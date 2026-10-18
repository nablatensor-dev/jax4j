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

import com.jax4j.tensor.*;
import java.util.Arrays;

/**
 * Small runtime diagnostic; --cpu proves explicit CPU startup does not initialize GPU providers.
 */
public final class WgpuInfo {
  private WgpuInfo() {}

  public static void main(String[] args) {
    try (Tensor x = Tensors.arrayOn(new float[] {1, 2}, Shape.of(2), Device.cpu());
        Tensor y = x.mul(2)) {
      if (!Arrays.equals(new float[] {2, 4}, y.toFloatArray()))
        throw new AssertionError("CPU startup failed");
    }
    if (args.length > 0 && args[0].equals("--cpu")) {
      System.out.println("CPU OK");
      return;
    }
    var backend = new WgpuBackend();
    boolean available = backend.isAvailable();
    if (!available) {
      System.out.println("wgpu unavailable: " + backend.unavailableReason());
      try {
        BackendRegistry.forDevice(Device.wgpu());
        throw new AssertionError("Explicit wgpu request unexpectedly succeeded");
      } catch (IllegalStateException expected) {
        System.out.println("Explicit wgpu request rejected; CPU remains available");
      }
      System.exit(2);
    }
    System.out.println(
        "wgpu-native 0x"
            + Integer.toHexString(backend.nativeVersion())
            + ": "
            + backend.deviceName());
  }
}
