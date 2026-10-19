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
package com.nablatensor.jax4j.tensor.spi;

import com.nablatensor.jax4j.tensor.*;
import java.lang.ref.Cleaner;
import java.util.concurrent.atomic.AtomicBoolean;

/** Native storage with deterministic and Cleaner-backed release through its owning runtime. */
public final class GpuBuffer implements DeviceBuffer {
  private static final Cleaner CLEANER = Cleaner.create();
  public final long buffer, memory;
  public final GpuRuntime runtime;
  private final Shape shape;
  private final DType dtype;
  private final Device device;
  private final AtomicBoolean closed = new AtomicBoolean();
  private final Cleaner.Cleanable cleanable;

  private record Free(GpuRuntime runtime, long buffer, long memory) implements Runnable {
    public void run() {
      runtime.free(buffer, memory);
    }
  }

  public GpuBuffer(
      GpuRuntime runtime, long buffer, long memory, Shape shape, DType dtype, Device device) {
    this.runtime = runtime;
    this.buffer = buffer;
    this.memory = memory;
    this.shape = shape;
    this.dtype = dtype;
    this.device = device;
    cleanable = CLEANER.register(this, new Free(runtime, buffer, memory));
  }

  public Shape shape() {
    return shape;
  }

  public DType dtype() {
    return dtype;
  }

  public Device device() {
    return device;
  }

  public int count() {
    return Math.toIntExact(shape.size());
  }

  public void requireOpen() {
    if (closed.get()) throw new IllegalStateException("GPU buffer is closed");
  }

  public void release() {
    if (closed.compareAndSet(false, true)) cleanable.clean();
  }
}
