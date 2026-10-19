/*
 * Copyright 2026 The NablaTensor Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nablatensor.jax4j.backend.wgpu;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.function.BooleanSupplier;

/**
 * Headless wgpu-native 29.0.1.1 runtime with batched command encoding and deferred destruction.
 * Cooperative kernels use WGSL; other native GLSL sources are validated/translated by Naga. No
 * Vulkan-module or shaderc dependency is loaded.
 */
final class WgpuRuntime {

  private WgpuRuntime() {}

  private static final Arena LIB = Arena.ofShared();
  private static final Linker LINKER = Linker.nativeLinker();

  // ---- symbol loading --------------------------------------------------

  private static final SymbolLookup WGPU = lookup();

  private static SymbolLookup lookup() {
    String explicit = System.getProperty("jax4j.wgpu.native_path");
    if (explicit == null || explicit.isBlank()) {
      explicit = System.getenv("JAX4J_WGPU_NATIVE_PATH");
    }
    if (explicit != null && !explicit.isBlank()) {
      return SymbolLookup.libraryLookup(explicit, LIB);
    }
    String xdg = System.getenv("XDG_CACHE_HOME");
    Path cacheBase =
        (xdg != null && !xdg.isBlank())
            ? Path.of(xdg)
            : Path.of(System.getProperty("user.home", "."), ".cache");
    String cached =
        cacheBase
            .resolve("nablatensor")
            .resolve("wgpu-native")
            .resolve("lib")
            .resolve("libwgpu_native.so")
            .toString();
    String[] candidates = {cached, "/usr/local/lib/libwgpu_native.so", "libwgpu_native.so"};
    for (String candidate : candidates) {
      try {
        return SymbolLookup.libraryLookup(candidate, LIB);
      } catch (RuntimeException ignored) {
        // try the next candidate
      }
    }
    return SymbolLookup.libraryLookup(candidates[candidates.length - 1], LIB);
  }

  private static MethodHandle wg(String name, FunctionDescriptor descriptor) {
    return LINKER.downcallHandle(
        WGPU.find(name).orElseThrow(() -> new UnsatisfiedLinkError(name)), descriptor);
  }

  // ---- by-value struct shapes (see webgpu.h) ----------------------------

  /**
   * {@code WGPUStringView}: {@code {char const * data; size_t length;}} — 16 bytes, both eightbytes
   * classified INTEGER, so this coarse layout is ABI- identical to the named struct for the
   * by-value callback parameter it appears in.
   */
  private static final MemoryLayout STRING_VIEW = MemoryLayout.structLayout(ADDRESS, JAVA_LONG);

  /**
   * The shape shared by {@code WGPURequestAdapterCallbackInfo}, {@code
   * WGPURequestDeviceCallbackInfo} and {@code WGPUBufferMapCallbackInfo}: {@code
   * {WGPUChainedStruct* next; WGPUCallbackMode mode; <pad4>; callback; userdata1; userdata2;}} — 40
   * bytes, passed by value (SysV MEMORY class, &gt;16 bytes), so only total size/alignment matter
   * to the calling convention.
   */
  private static final MemoryLayout CALLBACK_INFO =
      MemoryLayout.structLayout(
          ADDRESS, JAVA_INT, MemoryLayout.paddingLayout(4), ADDRESS, ADDRESS, ADDRESS);

  // ---- enum / flag constants (see webgpu.h, wgpu.h) ---------------------

  private static final int CALLBACK_MODE_ALLOW_PROCESS_EVENTS = 0x2;
  private static final int STYPE_SHADER_SOURCE_WGSL = 0x2;
  private static final long SHADER_STAGE_COMPUTE = 0x4L;
  private static final int BUFFER_BINDING_TYPE_STORAGE = 0x3;
  private static final int BUFFER_BINDING_TYPE_READ_ONLY_STORAGE = 0x4;
  private static final long BUFFER_USAGE_MAP_READ = 0x1L;
  private static final long BUFFER_USAGE_COPY_SRC = 0x4L;
  private static final long BUFFER_USAGE_COPY_DST = 0x8L;
  private static final long BUFFER_USAGE_STORAGE = 0x80L;
  private static final long MAP_MODE_READ = 0x1L;
  private static final int REQUEST_ADAPTER_STATUS_SUCCESS = 1;
  private static final int REQUEST_DEVICE_STATUS_SUCCESS = 1;
  private static final int MAP_ASYNC_STATUS_SUCCESS = 1;
  private static final long WGPU_WHOLE_SIZE = -1L; // UINT64_MAX
  private static final long WGPU_STRLEN = -1L; // UINT64_MAX: "use the null terminator"

  // ---- downcall handles ---------------------------------------------------

  private static final MethodHandle GET_VERSION =
      wg("wgpuGetVersion", FunctionDescriptor.of(JAVA_INT));
  private static final MethodHandle ADAPTER_GET_LIMITS =
      wg("wgpuAdapterGetLimits", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
  private static final MethodHandle SHADER_MODULE_RELEASE =
      wg("wgpuShaderModuleRelease", FunctionDescriptor.ofVoid(ADDRESS));
  private static final MethodHandle PIPELINE_LAYOUT_RELEASE =
      wg("wgpuPipelineLayoutRelease", FunctionDescriptor.ofVoid(ADDRESS));
  private static volatile String nativeError;

  private static void onError(
      MemorySegment devicePointer,
      int type,
      MemorySegment message,
      MemorySegment a,
      MemorySegment b) {
    nativeError = "wgpu error " + type + ": " + readStringView(message, 0);
  }

  private static final MemorySegment ERROR_CALLBACK_STUB =
      upcall(
          "onError",
          MethodType.methodType(
              void.class,
              MemorySegment.class,
              int.class,
              MemorySegment.class,
              MemorySegment.class,
              MemorySegment.class),
          FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, STRING_VIEW, ADDRESS, ADDRESS));

  private static void checkError() {
    String error = nativeError;
    if (error != null) {
      nativeError = null;
      throw new IllegalStateException(error);
    }
  }

  static int nativeVersion() {
    try {
      return (int) GET_VERSION.invoke();
    } catch (Throwable e) {
      throw new IllegalStateException(e);
    }
  }

  private static final MethodHandle CREATE_INSTANCE =
      wg("wgpuCreateInstance", FunctionDescriptor.of(ADDRESS, ADDRESS));
  private static final MethodHandle REQUEST_ADAPTER =
      wg(
          "wgpuInstanceRequestAdapter",
          FunctionDescriptor.of(JAVA_LONG, ADDRESS, ADDRESS, CALLBACK_INFO));
  private static final MethodHandle REQUEST_DEVICE =
      wg(
          "wgpuAdapterRequestDevice",
          FunctionDescriptor.of(JAVA_LONG, ADDRESS, ADDRESS, CALLBACK_INFO));
  private static final MethodHandle PROCESS_EVENTS =
      wg("wgpuInstanceProcessEvents", FunctionDescriptor.ofVoid(ADDRESS));
  private static final MethodHandle ADAPTER_GET_INFO =
      wg("wgpuAdapterGetInfo", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
  private static final MethodHandle DEVICE_GET_QUEUE =
      wg("wgpuDeviceGetQueue", FunctionDescriptor.of(ADDRESS, ADDRESS));
  private static final MethodHandle DEVICE_POLL =
      wg("wgpuDevicePoll", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS));

  private static final MethodHandle CREATE_BUFFER =
      wg("wgpuDeviceCreateBuffer", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle BUFFER_DESTROY =
      wg("wgpuBufferDestroy", FunctionDescriptor.ofVoid(ADDRESS));
  private static final MethodHandle BUFFER_RELEASE =
      wg("wgpuBufferRelease", FunctionDescriptor.ofVoid(ADDRESS));
  private static final MethodHandle BUFFER_MAP_ASYNC =
      wg(
          "wgpuBufferMapAsync",
          FunctionDescriptor.of(
              JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_LONG, CALLBACK_INFO));
  private static final MethodHandle BUFFER_GET_CONST_MAPPED_RANGE =
      wg(
          "wgpuBufferGetConstMappedRange",
          FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG));
  private static final MethodHandle BUFFER_UNMAP =
      wg("wgpuBufferUnmap", FunctionDescriptor.ofVoid(ADDRESS));

  private static final MethodHandle CREATE_SHADER_MODULE =
      wg("wgpuDeviceCreateShaderModule", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle CREATE_BIND_GROUP_LAYOUT =
      wg("wgpuDeviceCreateBindGroupLayout", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle CREATE_BIND_GROUP =
      wg("wgpuDeviceCreateBindGroup", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle CREATE_PIPELINE_LAYOUT =
      wg("wgpuDeviceCreatePipelineLayout", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle CREATE_COMPUTE_PIPELINE =
      wg("wgpuDeviceCreateComputePipeline", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle BIND_GROUP_RELEASE =
      wg("wgpuBindGroupRelease", FunctionDescriptor.ofVoid(ADDRESS));

  private static final MethodHandle CREATE_COMMAND_ENCODER =
      wg("wgpuDeviceCreateCommandEncoder", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle BEGIN_COMPUTE_PASS =
      wg("wgpuCommandEncoderBeginComputePass", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle COPY_BUFFER_TO_BUFFER =
      wg(
          "wgpuCommandEncoderCopyBufferToBuffer",
          FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_LONG));
  private static final MethodHandle COMMAND_ENCODER_FINISH =
      wg("wgpuCommandEncoderFinish", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS));
  private static final MethodHandle COMMAND_ENCODER_RELEASE =
      wg("wgpuCommandEncoderRelease", FunctionDescriptor.ofVoid(ADDRESS));
  private static final MethodHandle COMMAND_BUFFER_RELEASE =
      wg("wgpuCommandBufferRelease", FunctionDescriptor.ofVoid(ADDRESS));

  private static final MethodHandle COMPUTE_SET_PIPELINE =
      wg("wgpuComputePassEncoderSetPipeline", FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));
  private static final MethodHandle COMPUTE_SET_BIND_GROUP =
      wg(
          "wgpuComputePassEncoderSetBindGroup",
          FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
  private static final MethodHandle COMPUTE_DISPATCH =
      wg(
          "wgpuComputePassEncoderDispatchWorkgroups",
          FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT));
  private static final MethodHandle COMPUTE_PASS_END =
      wg("wgpuComputePassEncoderEnd", FunctionDescriptor.ofVoid(ADDRESS));
  private static final MethodHandle COMPUTE_PASS_RELEASE =
      wg("wgpuComputePassEncoderRelease", FunctionDescriptor.ofVoid(ADDRESS));

  private static final MethodHandle QUEUE_SUBMIT =
      wg("wgpuQueueSubmit", FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, ADDRESS));
  private static final MethodHandle QUEUE_WRITE_BUFFER =
      wg(
          "wgpuQueueWriteBuffer",
          FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG));

  // ---- upcall stubs (native -> Java callbacks) --------------------------

  private static volatile boolean adapterDone;
  private static volatile int adapterStatus;
  private static volatile MemorySegment adapterHandle;

  private static volatile boolean deviceDone;
  private static volatile int deviceStatus;
  private static volatile MemorySegment deviceHandle;

  private static volatile boolean mapDone;
  private static volatile int mapStatus;

  @SuppressWarnings("unused") // invoked by native code via the upcall stub
  private static void onRequestAdapter(
      int status,
      MemorySegment adapter,
      MemorySegment message,
      MemorySegment ud1,
      MemorySegment ud2) {
    adapterStatus = status;
    adapterHandle = adapter;
    adapterDone = true;
  }

  @SuppressWarnings("unused")
  private static void onRequestDevice(
      int status,
      MemorySegment device,
      MemorySegment message,
      MemorySegment ud1,
      MemorySegment ud2) {
    deviceStatus = status;
    deviceHandle = device;
    deviceDone = true;
  }

  @SuppressWarnings("unused")
  private static void onBufferMap(
      int status, MemorySegment message, MemorySegment ud1, MemorySegment ud2) {
    mapStatus = status;
    mapDone = true;
  }

  private static final MemorySegment ADAPTER_CALLBACK_STUB =
      upcall(
          "onRequestAdapter",
          MethodType.methodType(
              void.class,
              int.class,
              MemorySegment.class,
              MemorySegment.class,
              MemorySegment.class,
              MemorySegment.class),
          FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, STRING_VIEW, ADDRESS, ADDRESS));
  private static final MemorySegment DEVICE_CALLBACK_STUB =
      upcall(
          "onRequestDevice",
          MethodType.methodType(
              void.class,
              int.class,
              MemorySegment.class,
              MemorySegment.class,
              MemorySegment.class,
              MemorySegment.class),
          FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, STRING_VIEW, ADDRESS, ADDRESS));
  private static final MemorySegment MAP_CALLBACK_STUB =
      upcall(
          "onBufferMap",
          MethodType.methodType(
              void.class, int.class, MemorySegment.class, MemorySegment.class, MemorySegment.class),
          FunctionDescriptor.ofVoid(JAVA_INT, STRING_VIEW, ADDRESS, ADDRESS));

  private static MemorySegment upcall(
      String methodName, MethodType type, FunctionDescriptor descriptor) {
    try {
      MethodHandle target = MethodHandles.lookup().findStatic(WgpuRuntime.class, methodName, type);
      return LINKER.upcallStub(target, descriptor, LIB);
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  // ---- device state -------------------------------------------------------

  private static boolean ready;
  private static MemorySegment instance;
  private static MemorySegment adapter;
  private static MemorySegment device;
  private static MemorySegment queue;
  private static String deviceName = "";
  private static long readbackBuffer;
  private static long readbackCapacity;
  private static MemorySegment encoder = MemorySegment.NULL, pass = MemorySegment.NULL;
  private static int pendingDispatches;
  private static final java.util.List<Long> pendingFrees = new java.util.ArrayList<>();
  private static final Map<Long, Long> bufferSizes = new HashMap<>();
  private static final Map<String, Long> parameterBuffers = new HashMap<>();
  private static final Map<String, Pipeline> PIPELINES = new HashMap<>();

  private static final class Pipeline {
    final MemorySegment handle;
    final MemorySegment bindGroupLayout;
    final boolean parameters;
    private MemorySegment cachedBindGroup;
    private long[] cachedBuffers;

    Pipeline(MemorySegment handle, MemorySegment bindGroupLayout, boolean parameters) {
      this.parameters = parameters;
      this.handle = handle;
      this.bindGroupLayout = bindGroupLayout;
    }

    synchronized MemorySegment bindGroupFor(long[] buffers) throws Throwable {
      if (cachedBindGroup != null && Arrays.equals(cachedBuffers, buffers)) {
        return cachedBindGroup;
      }
      if (cachedBindGroup != null) {
        BIND_GROUP_RELEASE.invoke(cachedBindGroup);
      }
      cachedBindGroup = buildBindGroup(bindGroupLayout, buffers);
      cachedBuffers = buffers.clone();
      return cachedBindGroup;
    }
  }

  private static MemorySegment ptr(long address) {
    return MemorySegment.ofAddress(address);
  }

  static boolean probe() {
    try {
      init();
      return ready;
    } catch (Throwable failure) {
      return false;
    }
  }

  static String deviceName() {
    init();
    return deviceName;
  }

  static synchronized void init() {
    if (ready) {
      return;
    }
    try {
      if (nativeVersion() != 0x1d000101)
        throw new IllegalStateException(
            "Unsupported wgpu-native ABI: expected 29.0.1.1, got 0x"
                + Integer.toHexString(nativeVersion()));
      instance = (MemorySegment) CREATE_INSTANCE.invoke(MemorySegment.NULL);
      if (instance.equals(MemorySegment.NULL)) {
        throw new RuntimeException("wgpuCreateInstance returned NULL");
      }
      requestAdapter();
      requestDevice();
      queue = (MemorySegment) DEVICE_GET_QUEUE.invoke(device);
      readDeviceName();
      ready = true;
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  private static void requestAdapter() throws Throwable {
    adapterDone = false;
    adapterStatus = 0;
    adapterHandle = null;
    try (Arena a = Arena.ofConfined()) {
      MemorySegment cb = a.allocate(CALLBACK_INFO);
      cb.set(ADDRESS, 0, MemorySegment.NULL);
      cb.set(JAVA_INT, 8, CALLBACK_MODE_ALLOW_PROCESS_EVENTS);
      cb.set(ADDRESS, 16, ADAPTER_CALLBACK_STUB);
      cb.set(ADDRESS, 24, MemorySegment.NULL);
      cb.set(ADDRESS, 32, MemorySegment.NULL);
      MemorySegment options = a.allocate(32);
      options.set(JAVA_INT, 12, 2);
      options.set(JAVA_INT, 20, 6); // high-performance Vulkan adapter
      REQUEST_ADAPTER.invoke(instance, options, cb);
    }
    waitFor(() -> adapterDone, "adapter request");
    if (adapterStatus != REQUEST_ADAPTER_STATUS_SUCCESS
        || adapterHandle == null
        || adapterHandle.equals(MemorySegment.NULL)) {
      throw new RuntimeException("no WebGPU adapter available (status " + adapterStatus + ")");
    }
    adapter = adapterHandle;
  }

  private static void requestDevice() throws Throwable {
    deviceDone = false;
    deviceStatus = 0;
    deviceHandle = null;
    try (Arena a = Arena.ofConfined()) {
      MemorySegment cb = a.allocate(CALLBACK_INFO);
      cb.set(ADDRESS, 0, MemorySegment.NULL);
      cb.set(JAVA_INT, 8, CALLBACK_MODE_ALLOW_PROCESS_EVENTS);
      cb.set(ADDRESS, 16, DEVICE_CALLBACK_STUB);
      cb.set(ADDRESS, 24, MemorySegment.NULL);
      cb.set(ADDRESS, 32, MemorySegment.NULL);
      MemorySegment limits = a.allocate(152);
      if ((int) ADAPTER_GET_LIMITS.invoke(adapter, limits) != 1)
        throw new IllegalStateException("Could not query wgpu limits");
      MemorySegment descriptor = a.allocate(144);
      nullLabel(descriptor, 8);
      nullLabel(descriptor, 56);
      descriptor.set(ADDRESS, 40, limits);
      descriptor.set(JAVA_INT, 80, CALLBACK_MODE_ALLOW_PROCESS_EVENTS);
      descriptor.set(ADDRESS, 88, ERROR_CALLBACK_STUB);
      descriptor.set(ADDRESS, 120, ERROR_CALLBACK_STUB);
      REQUEST_DEVICE.invoke(adapter, descriptor, cb);
    }
    waitFor(() -> deviceDone, "device request");
    if (deviceStatus != REQUEST_DEVICE_STATUS_SUCCESS
        || deviceHandle == null
        || deviceHandle.equals(MemorySegment.NULL)) {
      throw new RuntimeException("no WebGPU device available (status " + deviceStatus + ")");
    }
    device = deviceHandle;
  }

  private static void waitFor(BooleanSupplier done, String what) throws Throwable {
    long deadline = System.nanoTime() + 5_000_000_000L;
    while (!done.getAsBoolean()) {
      PROCESS_EVENTS.invoke(instance);
      if (done.getAsBoolean()) {
        return;
      }
      if (System.nanoTime() > deadline) {
        throw new RuntimeException("timed out waiting for " + what);
      }
      Thread.sleep(1);
    }
  }

  private static void readDeviceName() {
    try (Arena a = Arena.ofConfined()) {
      MemorySegment info = a.allocate(96);
      int status = (int) ADAPTER_GET_INFO.invoke(adapter, info);
      if (status != 1) {
        deviceName = "wgpu device";
        return;
      }
      String dev = readStringView(info, 40);
      String desc = readStringView(info, 56);
      int backendType = info.get(JAVA_INT, 72);
      String backend =
          switch (backendType) {
            case 3 -> "D3D11";
            case 4 -> "D3D12";
            case 5 -> "Metal";
            case 6 -> "Vulkan";
            case 7 -> "OpenGL";
            case 8 -> "OpenGLES";
            default -> "backend " + backendType;
          };
      String name = !dev.isBlank() ? dev : !desc.isBlank() ? desc : "wgpu device";
      deviceName = name + " (" + backend + ")";
    } catch (Throwable failure) {
      deviceName = "wgpu device";
    }
  }

  /** Reads a {@code WGPUStringView} embedded at {@code offset} within {@code struct}. */
  private static String readStringView(MemorySegment struct, long offset) {
    MemorySegment data = struct.get(ADDRESS, offset);
    long length = struct.get(JAVA_LONG, offset + 8);
    if (data.equals(MemorySegment.NULL) || length <= 0) {
      return "";
    }
    byte[] bytes = data.reinterpret(length).toArray(JAVA_BYTE);
    return new String(bytes, StandardCharsets.UTF_8);
  }

  // ---- shader / pipeline registration ------------------------------------

  /**
   * Builds a native GLSL compute pipeline with explicit storage bindings. Push constants become a
   * read-only parameter binding. Sources are validated by Naga.
   */
  static synchronized void registerPipeline(String name, String source, int bindings) {
    init();
    if (PIPELINES.containsKey(name)) {
      return;
    }
    try {
      boolean parameters = source.contains("layout(push_constant)");
      int n = bindings + (parameters ? 1 : 0);
      boolean[] readOnly = new boolean[n];
      Arrays.fill(readOnly, true);
      readOnly[0] = false;
      String glsl =
          source
              .replace("writeonly", "")
              .replace("layout(push_constant)", "layout(std430,set=0,binding=" + bindings + ")");
      glsl = glsl.replace(" uniform P ", " readonly buffer P ");
      // GLSL's extended multiply is not implemented by Naga's GLSL frontend.
      glsl =
          glsl.replace(
              "uint hi,lo;umulExtended(a.x,b.x,hi,lo);",
              "uint lo=a.x*b.x;uint a0=a.x&65535u,a1=a.x>>16u,b0=b.x&65535u,b1=b.x>>16u;uint"
                  + " w0=a0*b0;uint t=a1*b0+(w0>>16u);uint w1=t&65535u,w2=t>>16u;w1+=a0*b1;uint"
                  + " hi=a1*b1+w2+(w1>>16u);");

      try (Arena a = Arena.ofConfined()) {
        String wgsl = WgpuShaders.source(name);
        byte[] sourceBytes = (wgsl == null ? glsl : wgsl).getBytes(StandardCharsets.UTF_8);
        MemorySegment sourceMemory = a.allocate(sourceBytes.length);
        MemorySegment.copy(sourceBytes, 0, sourceMemory, JAVA_BYTE, 0, sourceBytes.length);

        MemorySegment chain = a.allocate(wgsl == null ? 56 : 32);
        chain.set(JAVA_INT, 8, wgsl == null ? 0x00030003 : STYPE_SHADER_SOURCE_WGSL);
        if (wgsl == null) {
          chain.set(JAVA_LONG, 16, SHADER_STAGE_COMPUTE);
          chain.set(ADDRESS, 24, sourceMemory);
          chain.set(JAVA_LONG, 32, sourceBytes.length);
        } else {
          chain.set(ADDRESS, 16, sourceMemory);
          chain.set(JAVA_LONG, 24, sourceBytes.length);
        }

        // WGPUShaderModuleDescriptor: {nextInChain8, label:StringView16} = 24 bytes
        MemorySegment smDesc = a.allocate(24);
        smDesc.set(ADDRESS, 0, chain);
        nullLabel(smDesc, 8);
        MemorySegment shaderModule = (MemorySegment) CREATE_SHADER_MODULE.invoke(device, smDesc);
        checkError();
        if (shaderModule.equals(MemorySegment.NULL)) {
          throw new RuntimeException(
              "wgpuDeviceCreateShaderModule failed for kernel " + name + "\n--- GLSL ---\n" + glsl);
        }

        // WGPUBindGroupLayoutEntry, 120 bytes each (see webgpu.h for the field-by-field
        // derivation).
        long entrySize = 120;
        MemorySegment entries = a.allocate((long) n * entrySize);
        for (int i = 0; i < n; i++) {
          long base = (long) i * entrySize;
          entries.set(JAVA_INT, base + 8, i); // binding
          entries.set(JAVA_LONG, base + 16, SHADER_STAGE_COMPUTE); // visibility
          entries.set(
              JAVA_INT,
              base + 40, // buffer.type
              readOnly[i] ? BUFFER_BINDING_TYPE_READ_ONLY_STORAGE : BUFFER_BINDING_TYPE_STORAGE);
          // sampler/texture/storageTexture stay zero => BindingNotUsed
        }
        MemorySegment bglDesc = a.allocate(40); // {nextInChain8, label16, entryCount8, entries8}
        nullLabel(bglDesc, 8);
        bglDesc.set(JAVA_LONG, 24, n);
        bglDesc.set(ADDRESS, 32, entries);
        MemorySegment bgl = (MemorySegment) CREATE_BIND_GROUP_LAYOUT.invoke(device, bglDesc);

        MemorySegment bglHandle = a.allocate(ADDRESS);
        bglHandle.set(ADDRESS, 0, bgl);
        MemorySegment plDesc =
            a.allocate(48); // {nextInChain8, label16, count8, ptr8, immediateSize4+pad4}
        nullLabel(plDesc, 8);
        plDesc.set(JAVA_LONG, 24, 1L);
        plDesc.set(ADDRESS, 32, bglHandle);
        MemorySegment pl = (MemorySegment) CREATE_PIPELINE_LAYOUT.invoke(device, plDesc);

        byte[] entryBytes = "main".getBytes(StandardCharsets.UTF_8);
        MemorySegment entryPoint = a.allocate(entryBytes.length);
        MemorySegment.copy(entryBytes, 0, entryPoint, JAVA_BYTE, 0, entryBytes.length);

        // WGPUComputePipelineDescriptor: {nextInChain8, label16, layout8,
        // compute:{next8,module8,entryPoint16,constCount8,constants8}=48} = 80
        MemorySegment cpDesc = a.allocate(80);
        nullLabel(cpDesc, 8);
        cpDesc.set(ADDRESS, 24, pl);
        cpDesc.set(ADDRESS, 40, shaderModule);
        cpDesc.set(ADDRESS, 48, entryPoint);
        cpDesc.set(JAVA_LONG, 56, entryBytes.length);
        MemorySegment pipeline = (MemorySegment) CREATE_COMPUTE_PIPELINE.invoke(device, cpDesc);
        checkError();
        if (pipeline.equals(MemorySegment.NULL)) {
          throw new RuntimeException("wgpuDeviceCreateComputePipeline failed for kernel " + name);
        }
        PIPELINES.put(name, new Pipeline(pipeline, bgl, parameters));
        SHADER_MODULE_RELEASE.invoke(shaderModule);
        PIPELINE_LAYOUT_RELEASE.invoke(pl);
        WgpuDiagnostics.compiled();
      }
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  /**
   * Sets a {@code WGPUStringView} field at {@code offset} to the null value ({@code {NULL,
   * SIZE_MAX}}).
   */
  private static void nullLabel(MemorySegment struct, long offset) {
    struct.set(ADDRESS, offset, MemorySegment.NULL);
    struct.set(JAVA_LONG, offset + 8, WGPU_STRLEN);
  }

  private static MemorySegment buildBindGroup(MemorySegment bindGroupLayout, long[] buffers)
      throws Throwable {
    int n = buffers.length;
    try (Arena a = Arena.ofConfined()) {
      // WGPUBindGroupEntry, 56 bytes each: {next8, binding4+pad4, buffer8, offset8, size8,
      // sampler8, textureView8}
      MemorySegment entries = a.allocate((long) n * 56);
      for (int i = 0; i < n; i++) {
        long base = (long) i * 56;
        entries.set(JAVA_INT, base + 8, i);
        entries.set(ADDRESS, base + 16, ptr(buffers[i]));
        entries.set(JAVA_LONG, base + 32, WGPU_WHOLE_SIZE);
      }
      MemorySegment desc =
          a.allocate(48); // {nextInChain8, label16, layout8, entryCount8, entries8}
      nullLabel(desc, 8);
      desc.set(ADDRESS, 24, bindGroupLayout);
      desc.set(JAVA_LONG, 32, n);
      desc.set(ADDRESS, 40, entries);
      MemorySegment bindGroup = (MemorySegment) CREATE_BIND_GROUP.invoke(device, desc);
      if (bindGroup.equals(MemorySegment.NULL)) {
        throw new RuntimeException("wgpuDeviceCreateBindGroup failed");
      }
      return bindGroup;
    }
  }

  // ---- buffers ------------------------------------------------------------

  static synchronized long alloc(long bytes) {
    init();
    try {
      long buffer =
          createBuffer(bytes, BUFFER_USAGE_STORAGE | BUFFER_USAGE_COPY_SRC | BUFFER_USAGE_COPY_DST);
      bufferSizes.put(buffer, bytes);
      WgpuDiagnostics.allocated(bytes);
      return buffer;
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  private static long createBuffer(long bytes, long usage) throws Throwable {
    try (Arena a = Arena.ofConfined()) {
      MemorySegment desc = a.allocate(48);
      nullLabel(desc, 8);
      desc.set(JAVA_LONG, 24, usage);
      desc.set(JAVA_LONG, 32, bytes);
      MemorySegment buf = (MemorySegment) CREATE_BUFFER.invoke(device, desc);
      if (buf.equals(MemorySegment.NULL)) {
        throw new RuntimeException("wgpuDeviceCreateBuffer failed for " + bytes + " bytes");
      }
      return buf.address();
    }
  }

  static synchronized void free(long buffer) {
    if (pendingDispatches > 0) {
      pendingFrees.add(buffer);
      return;
    }
    freeNow(buffer);
  }

  private static void freeNow(long buffer) {
    try {
      for (Pipeline pipeline : PIPELINES.values())
        if (pipeline.cachedBuffers != null)
          for (long used : pipeline.cachedBuffers)
            if (used == buffer) {
              BIND_GROUP_RELEASE.invoke(pipeline.cachedBindGroup);
              pipeline.cachedBindGroup = null;
              pipeline.cachedBuffers = null;
              break;
            }
      BUFFER_DESTROY.invoke(ptr(buffer));
      BUFFER_RELEASE.invoke(ptr(buffer));
      Long bytes = bufferSizes.remove(buffer);
      if (bytes != null) WgpuDiagnostics.freed(bytes);
    } catch (Throwable failure) {
      throw new IllegalStateException(failure);
    }
  }

  static synchronized void writeFloats(long buffer, float[] data) {
    init();
    sync();
    WgpuDiagnostics.uploaded((long) data.length * Float.BYTES);
    try (Arena a = Arena.ofConfined()) {
      MemorySegment seg = a.allocate((long) data.length * Float.BYTES);
      MemorySegment.copy(data, 0, seg, JAVA_FLOAT, 0, data.length);
      QUEUE_WRITE_BUFFER.invoke(queue, ptr(buffer), 0L, seg, (long) data.length * Float.BYTES);
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  static synchronized float[] readFloats(long buffer, int count) {
    init();
    long bytes = (long) count * Float.BYTES;
    if (count == 0) return new float[0];
    sync();
    WgpuDiagnostics.downloaded(bytes);
    try {
      ensureReadback(bytes);
      try (Arena a = Arena.ofConfined()) {
        MemorySegment encoder =
            (MemorySegment) CREATE_COMMAND_ENCODER.invoke(device, MemorySegment.NULL);
        COPY_BUFFER_TO_BUFFER.invoke(encoder, ptr(buffer), 0L, ptr(readbackBuffer), 0L, bytes);
        MemorySegment cmd =
            (MemorySegment) COMMAND_ENCODER_FINISH.invoke(encoder, MemorySegment.NULL);
        MemorySegment cmdHandle = a.allocate(ADDRESS);
        cmdHandle.set(ADDRESS, 0, cmd);
        QUEUE_SUBMIT.invoke(queue, 1L, cmdHandle);
        DEVICE_POLL.invoke(device, 1, MemorySegment.NULL);
        COMMAND_BUFFER_RELEASE.invoke(cmd);
        COMMAND_ENCODER_RELEASE.invoke(encoder);

        mapDone = false;
        mapStatus = 0;
        MemorySegment cb = a.allocate(CALLBACK_INFO);
        cb.set(ADDRESS, 0, MemorySegment.NULL);
        cb.set(JAVA_INT, 8, CALLBACK_MODE_ALLOW_PROCESS_EVENTS);
        cb.set(ADDRESS, 16, MAP_CALLBACK_STUB);
        cb.set(ADDRESS, 24, MemorySegment.NULL);
        cb.set(ADDRESS, 32, MemorySegment.NULL);
        BUFFER_MAP_ASYNC.invoke(ptr(readbackBuffer), MAP_MODE_READ, 0L, bytes, cb);
        waitFor(() -> mapDone, "buffer map");
        if (mapStatus != MAP_ASYNC_STATUS_SUCCESS) {
          throw new RuntimeException("wgpuBufferMapAsync failed: status " + mapStatus);
        }

        MemorySegment mapped =
            ((MemorySegment) BUFFER_GET_CONST_MAPPED_RANGE.invoke(ptr(readbackBuffer), 0L, bytes))
                .reinterpret(bytes);
        float[] out = new float[count];
        MemorySegment.copy(mapped, JAVA_FLOAT, 0, out, 0, count);
        BUFFER_UNMAP.invoke(ptr(readbackBuffer));
        return out;
      }
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException runtime ? runtime : new RuntimeException(failure);
    }
  }

  private static void ensureReadback(long bytes) throws Throwable {
    if (readbackBuffer != 0 && readbackCapacity >= bytes) {
      return;
    }
    if (readbackBuffer != 0) {
      BUFFER_DESTROY.invoke(ptr(readbackBuffer));
      BUFFER_RELEASE.invoke(ptr(readbackBuffer));
    }
    readbackBuffer = createBuffer(bytes, BUFFER_USAGE_MAP_READ | BUFFER_USAGE_COPY_DST);
    readbackCapacity = bytes;
  }

  // ---- dispatch -------------------------------------------------------------

  /**
   * Encodes commands; sync/readback or a bounded batch submits and waits. Parameter buffers are
   * immutable.
   */
  static synchronized void dispatch(
      String kernel,
      int groupsX,
      int groupsY,
      int groupsZ,
      long[] tensors,
      int[] pushInts,
      int floatSlot,
      float floatValue) {
    init();
    Pipeline pipeline = PIPELINES.get(kernel);
    if (pipeline == null) throw new IllegalStateException("Unknown wgpu kernel: " + kernel);
    try {
      long[] buffers = tensors;
      if (pipeline.parameters) {
        int[] words = new int[32];
        if (pushInts != null) System.arraycopy(pushInts, 0, words, 0, pushInts.length);
        if (floatSlot >= 0) words[floatSlot] = Float.floatToRawIntBits(floatValue);
        String key = Arrays.toString(words);
        Long parameter = parameterBuffers.get(key);
        if (parameter == null) {
          if (parameterBuffers.size() >= 1024) {
            sync();
            for (long old : parameterBuffers.values()) freeNow(old);
            parameterBuffers.clear();
          }
          parameter = createBuffer(128, BUFFER_USAGE_STORAGE | BUFFER_USAGE_COPY_DST);
          try (Arena a = Arena.ofConfined()) {
            MemorySegment data = a.allocate(128);
            MemorySegment.copy(words, 0, data, JAVA_INT, 0, 32);
            QUEUE_WRITE_BUFFER.invoke(queue, ptr(parameter), 0L, data, 128L);
          }
          parameterBuffers.put(key, parameter);
          WgpuDiagnostics.parametersUploaded(128);
        }
        buffers = Arrays.copyOf(tensors, tensors.length + 1);
        buffers[tensors.length] = parameter;
      }
      MemorySegment group = pipeline.bindGroupFor(buffers);
      if (encoder.equals(MemorySegment.NULL)) {
        encoder = (MemorySegment) CREATE_COMMAND_ENCODER.invoke(device, MemorySegment.NULL);
        pass = (MemorySegment) BEGIN_COMPUTE_PASS.invoke(encoder, MemorySegment.NULL);
      }
      COMPUTE_SET_PIPELINE.invoke(pass, pipeline.handle);
      COMPUTE_SET_BIND_GROUP.invoke(pass, 0, group, 0L, MemorySegment.NULL);
      COMPUTE_DISPATCH.invoke(pass, groupsX, groupsY, groupsZ);
      checkError();
      WgpuDiagnostics.dispatched();
      if (++pendingDispatches >= 128) sync();
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException r ? r : new IllegalStateException(failure);
    }
  }

  static synchronized void sync() {
    if (!ready) return;
    try {
      if (pendingDispatches > 0) {
        COMPUTE_PASS_END.invoke(pass);
        MemorySegment command =
            (MemorySegment) COMMAND_ENCODER_FINISH.invoke(encoder, MemorySegment.NULL);
        checkError();
        try (Arena a = Arena.ofConfined()) {
          MemorySegment handle = a.allocate(ADDRESS);
          handle.set(ADDRESS, 0, command);
          QUEUE_SUBMIT.invoke(queue, 1L, handle);
        }
        DEVICE_POLL.invoke(device, 1, MemorySegment.NULL);
        checkError();
        COMPUTE_PASS_RELEASE.invoke(pass);
        COMMAND_BUFFER_RELEASE.invoke(command);
        COMMAND_ENCODER_RELEASE.invoke(encoder);
        pass = MemorySegment.NULL;
        encoder = MemorySegment.NULL;
        pendingDispatches = 0;
        for (long buffer : pendingFrees) freeNow(buffer);
        pendingFrees.clear();
      } else {
        DEVICE_POLL.invoke(device, 1, MemorySegment.NULL);
        checkError();
      }
    } catch (Throwable failure) {
      throw failure instanceof RuntimeException r ? r : new IllegalStateException(failure);
    }
  }
}
