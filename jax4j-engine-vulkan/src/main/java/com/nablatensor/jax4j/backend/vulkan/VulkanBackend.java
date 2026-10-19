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

import com.nablatensor.jax4j.tensor.DeviceType;
import com.nablatensor.jax4j.tensor.spi.*;

/** Vulkan tensor execution using the shared GPU kernels and native Vulkan runtime. */
public final class VulkanBackend extends GpuTensorBackend {
  protected GpuRuntime runtime() {
    return VulkanGpuRuntime.INSTANCE;
  }

  public String name() {
    return "vulkan";
  }

  public DeviceType deviceType() {
    return DeviceType.VULKAN;
  }

  public int priority() {
    return 40;
  }

  public boolean isAvailable() {
    try {
      return VulkanRuntime.probe();
    } catch (Throwable failure) {
      return false;
    }
  }

  public VulkanDiagnostics.Snapshot diagnostics() {
    return VulkanDiagnostics.snapshot();
  }
}
