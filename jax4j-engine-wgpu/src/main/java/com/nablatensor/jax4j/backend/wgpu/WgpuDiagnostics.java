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

/**
 * Tensor data transfers are separate from immutable dispatch-parameter uploads and staging
 * capacity.
 */
public final class WgpuDiagnostics {
  private static long dispatches,
      allocations,
      allocatedBytes,
      uploads,
      downloads,
      compilations,
      live,
      peak,
      parameters;

  private WgpuDiagnostics() {}

  public record Snapshot(
      long dispatches,
      long allocations,
      long allocatedBytes,
      long hostToDeviceBytes,
      long deviceToHostBytes,
      long compilations,
      long liveDeviceBytes,
      long peakDeviceBytes,
      long parameterUploadBytes) {
    public Snapshot minus(Snapshot b) {
      return new Snapshot(
          dispatches - b.dispatches,
          allocations - b.allocations,
          allocatedBytes - b.allocatedBytes,
          hostToDeviceBytes - b.hostToDeviceBytes,
          deviceToHostBytes - b.deviceToHostBytes,
          compilations - b.compilations,
          liveDeviceBytes - b.liveDeviceBytes,
          peakDeviceBytes,
          parameterUploadBytes - b.parameterUploadBytes);
    }
  }

  public static synchronized Snapshot snapshot() {
    return new Snapshot(
        dispatches,
        allocations,
        allocatedBytes,
        uploads,
        downloads,
        compilations,
        live,
        peak,
        parameters);
  }

  static synchronized void compiled() {
    compilations++;
  }

  static synchronized void dispatched() {
    dispatches++;
  }

  static synchronized void uploaded(long n) {
    uploads += n;
  }

  static synchronized void downloaded(long n) {
    downloads += n;
  }

  static synchronized void parametersUploaded(long n) {
    parameters += n;
  }

  static synchronized void allocated(long n) {
    allocations++;
    allocatedBytes += n;
    live += n;
    peak = Math.max(peak, live);
  }

  static synchronized void freed(long n) {
    live -= n;
  }
}
