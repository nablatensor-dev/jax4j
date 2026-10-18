package com.jax4j.tensor;

import com.jax4j.tensor.spi.ComputeBackend;
import com.jax4j.tensor.spi.DeviceBuffer;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/** Backend-neutral sequence of fused kernels and explicit fusion boundaries. */
final class ExecutionPlan {
  private record Binding(DeviceBuffer borrowed) implements DeviceBuffer {
    public Shape shape() {
      return borrowed.shape();
    }

    public DType dtype() {
      return borrowed.dtype();
    }

    public Device device() {
      return borrowed.device();
    }
  }

  private record Call(Method method, Object[] arguments, DeviceBuffer result) {}

  record Capture(ExecutionPlan plan, Object output) {}

  private final ComputeBackend backend;
  private final List<DeviceBuffer> inputMarkers;
  private final List<DeviceBuffer> outputMarkers;
  private final com.jax4j.tensor.tree.TreeDef outputDef;
  private final List<Call> calls;

  private ExecutionPlan(
      ComputeBackend backend, List<Tensor> inputs, Object output, List<Call> calls) {
    this.backend = backend;
    inputMarkers = inputs.stream().map(Tensor::buffer).toList();
    var flat = com.jax4j.tensor.tree.TreeUtil.flatten(output);
    outputDef = flat.def();
    outputMarkers = flat.leaves().stream().map(x -> ((Tensor) x).buffer()).toList();
    this.calls = List.copyOf(calls);
  }

  static Capture capture(
      java.util.function.Function<List<Tensor>, Object> body, List<Tensor> inputs) {
    ComputeBackend backend = inputs.get(0).backend();
    List<Call> calls = new ArrayList<>();
    ComputeBackend recording =
        (ComputeBackend)
            Proxy.newProxyInstance(
                ComputeBackend.class.getClassLoader(),
                new Class<?>[] {com.jax4j.tensor.spi.TracingBackend.class},
                (proxy, method, args) -> {
                  if (method.getName().equals("parentBackend")) return backend;
                  if (method.getName().equals("download")) {
                    throw new IllegalStateException(
                        "Host reads are not allowed in a compiled JIT function");
                  }
                  Object result = invoke(backend, method, unwrapArguments(args));
                  if (result instanceof DeviceBuffer
                      || method.getName().equals("release")
                      || method.getName().equals("sync")) {
                    calls.add(
                        new Call(
                            method,
                            copyArguments(args),
                            result instanceof DeviceBuffer b ? b : null));
                  }
                  return result;
                });
    List<Tensor> recordingInputs =
        inputs.stream().map(x -> new Tensor(recording, new Binding(x.buffer()))).toList();
    Object output = Jit.applyTree(body, recordingInputs, backend);
    ExecutionPlan plan = new ExecutionPlan(backend, recordingInputs, output, calls);
    var flat = com.jax4j.tensor.tree.TreeUtil.flatten(output);
    var leaves =
        flat.leaves().stream()
            .map(
                x -> {
                  Tensor tensor = (Tensor) x;
                  if (tensor.buffer() instanceof Binding binding) {
                    for (Tensor original : inputs)
                      if (original.buffer() == binding.borrowed()) return original;
                  }
                  return tensor;
                })
            .toList();
    return new Capture(plan, com.jax4j.tensor.tree.TreeUtil.unflatten(flat.def(), leaves));
  }

  Object execute(List<Tensor> inputs) {
    Map<DeviceBuffer, DeviceBuffer> buffers = new IdentityHashMap<>();
    Map<DeviceBuffer, Boolean> owned = new IdentityHashMap<>();
    for (int i = 0; i < inputs.size(); i++)
      buffers.put(inputMarkers.get(i), inputs.get(i).buffer());
    try {
      for (Call call : calls) {
        Object[] args = copyArguments(call.arguments());
        for (int i = 0; i < args.length; i++) {
          if (args[i] instanceof DeviceBuffer b) args[i] = buffers.getOrDefault(b, b);
          else if (args[i] instanceof DeviceBuffer[] array) {
            for (int j = 0; j < array.length; j++)
              array[j] = buffers.getOrDefault(array[j], array[j]);
          }
        }
        Object result = invoke(backend, call.method(), args);
        if (call.result() != null) {
          DeviceBuffer buffer = (DeviceBuffer) result;
          buffers.put(call.result(), buffer);
          // Backends may return an operand for identity broadcasts.
          if (inputs.stream().noneMatch(x -> x.buffer() == buffer) && !isOperand(buffer, args))
            owned.put(buffer, true);
        } else if (call.method().getName().equals("release")) owned.remove(args[0]);
      }
      List<Tensor> outputs = new ArrayList<>();
      Map<DeviceBuffer, Tensor> wrappers = new IdentityHashMap<>();
      for (Tensor input : inputs) wrappers.put(input.buffer(), input);
      for (DeviceBuffer marker : outputMarkers) {
        DeviceBuffer output = buffers.getOrDefault(marker, marker);
        owned.remove(output);
        outputs.add(wrappers.computeIfAbsent(output, b -> new Tensor(backend, b)));
      }
      return com.jax4j.tensor.tree.TreeUtil.unflatten(outputDef, outputs);
    } finally {
      for (DeviceBuffer buffer : owned.keySet()) backend.release(buffer);
    }
  }

  private static boolean isOperand(DeviceBuffer buffer, Object[] args) {
    for (Object arg : args) {
      if (arg == buffer) return true;
      if (arg instanceof DeviceBuffer[] array)
        for (DeviceBuffer b : array) if (b == buffer) return true;
    }
    return false;
  }

  private static Object[] unwrapArguments(Object[] args) {
    Object[] unwrapped = copyArguments(args);
    for (int i = 0; i < unwrapped.length; i++) {
      if (unwrapped[i] instanceof Binding b) unwrapped[i] = b.borrowed();
      else if (unwrapped[i] instanceof DeviceBuffer[] array)
        for (int j = 0; j < array.length; j++)
          if (array[j] instanceof Binding b) array[j] = b.borrowed();
    }
    return unwrapped;
  }

  private static Object[] copyArguments(Object[] args) {
    if (args == null) return new Object[0];
    Object[] copy = args.clone();
    for (int i = 0; i < copy.length; i++) {
      if (copy[i] instanceof DeviceBuffer[] a) copy[i] = a.clone();
      else if (copy[i] instanceof float[] a) copy[i] = a.clone();
      else if (copy[i] instanceof int[] a) copy[i] = a.clone();
    }
    return copy;
  }

  private static Object invoke(ComputeBackend backend, Method method, Object[] args) {
    try {
      return method.invoke(backend, args);
    } catch (InvocationTargetException e) {
      if (e.getCause() instanceof RuntimeException r) throw r;
      if (e.getCause() instanceof Error error) throw error;
      throw new IllegalStateException(e.getCause());
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    }
  }
}
