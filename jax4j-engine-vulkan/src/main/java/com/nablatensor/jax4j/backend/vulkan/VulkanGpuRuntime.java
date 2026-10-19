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
package com.nablatensor.jax4j.backend.vulkan;

import com.nablatensor.jax4j.tensor.spi.GpuRuntime;

enum VulkanGpuRuntime implements GpuRuntime {
  INSTANCE;

  public void init() {
    VulkanRuntime.init();
  }

  public String deviceName() {
    return VulkanRuntime.deviceName();
  }

  public void registerPipeline(String name, String source, int bindings) {
    VulkanRuntime.registerPipeline(name, source, bindings);
  }

  public long[] alloc(long bytes) {
    return VulkanRuntime.alloc(bytes);
  }

  public void free(long buffer, long memory) {
    VulkanRuntime.free(buffer, memory);
  }

  public void writeFloats(long memory, float[] values) {
    VulkanRuntime.writeFloats(memory, values);
  }

  public float[] readFloats(long memory, int count) {
    return VulkanRuntime.readFloats(memory, count);
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
    VulkanRuntime.dispatch(kernel, x, y, z, buffers, parameters, floatSlot, floatValue);
  }

  public void sync() {
    VulkanRuntime.sync();
  }
}
