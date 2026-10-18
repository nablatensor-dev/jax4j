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
package com.jax4j.backend.wgpu;

import com.jax4j.tensor.DeviceType;
import com.jax4j.tensor.spi.*;

/** wgpu-native tensor backend; runtime discovery is lazy and GPU fallback is never implicit. */
public final class WgpuBackend extends GpuTensorBackend {
  private volatile String unavailableReason = "not probed";

  protected GpuRuntime runtime() {
    return WgpuGpuRuntime.INSTANCE;
  }

  public String name() {
    return "wgpu";
  }

  public DeviceType deviceType() {
    return DeviceType.WGPU;
  }

  public int priority() {
    return 35;
  }

  public boolean isAvailable() {
    try {
      WgpuRuntime.init();
      unavailableReason = "";
      return true;
    } catch (Throwable failure) {
      Throwable root = failure;
      while (root.getCause() != null && root.getCause() != root) root = root.getCause();
      unavailableReason = root.toString();
      return false;
    }
  }

  public String unavailableReason() {
    return unavailableReason;
  }

  public int nativeVersion() {
    return WgpuRuntime.nativeVersion();
  }

  public WgpuDiagnostics.Snapshot diagnostics() {
    return WgpuDiagnostics.snapshot();
  }
}
