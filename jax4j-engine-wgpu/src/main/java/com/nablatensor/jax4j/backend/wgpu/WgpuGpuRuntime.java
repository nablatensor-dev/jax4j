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

import com.nablatensor.jax4j.tensor.spi.GpuRuntime;

enum WgpuGpuRuntime implements GpuRuntime {
  INSTANCE;

  public void init() {
    WgpuRuntime.init();
  }

  public String deviceName() {
    return WgpuRuntime.deviceName();
  }

  public void registerPipeline(String name, String source, int bindings) {
    WgpuRuntime.registerPipeline(name, source, bindings);
  }

  public long[] alloc(long bytes) {
    long handle = WgpuRuntime.alloc(bytes);
    return new long[] {handle, handle};
  }

  public void free(long buffer, long memory) {
    WgpuRuntime.free(buffer);
  }

  public void writeFloats(long memory, float[] values) {
    WgpuRuntime.writeFloats(memory, values);
  }

  public float[] readFloats(long memory, int count) {
    return WgpuRuntime.readFloats(memory, count);
  }

  public void dispatch(
      String kernel,
      int x,
      int y,
      int z,
      long[] buffers,
      int[] parameters,
      int floatSlot,
      float floatValue) {
    WgpuRuntime.dispatch(kernel, x, y, z, buffers, parameters, floatSlot, floatValue);
  }

  public void sync() {
    WgpuRuntime.sync();
  }
}
