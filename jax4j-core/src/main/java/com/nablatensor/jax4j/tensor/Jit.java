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
package com.nablatensor.jax4j.tensor;

import com.nablatensor.jax4j.tensor.spi.ComputeBackend;
import com.nablatensor.jax4j.tensor.spi.DeviceBuffer;

import java.util.function.UnaryOperator;

/**
 * Traces a fluent {@code Tensor} function and fuses each maximal elementwise
 * chain (add/sub/mul/div/unary/scalar ops) between matmul/transpose
 * boundaries into a single backend kernel launch, instead of one launch per
 * op. Tracing itself is cheap (plain object graph building); backends cache
 * compiled fused kernels by their generated source, so repeated calls with
 * the same op chain and shape reuse the compiled kernel.
 */
public final class Jit {

  private Jit() {
  }

  public static Tensor apply(UnaryOperator<Tensor> body, Tensor input) {
    return apply(xs -> body.apply(xs.get(0)), java.util.List.of(input));
  }

  /** Captures all arguments so layers, parameters and optimizer state remain fusible. */
  public static Tensor apply(java.util.function.Function<java.util.List<Tensor>, Tensor> body,
                             java.util.List<Tensor> inputs) {
    if (inputs.isEmpty()) throw new IllegalArgumentException("JIT requires inputs");
    ComputeBackend real = inputs.get(0).backend();
    for (Tensor input : inputs) {
      if (input.backend() != real || !input.device().equals(inputs.get(0).device()))
        throw new IllegalArgumentException("JIT inputs must share backend and device");
    }
    FusingBackend fusing = new FusingBackend(real);
    DeviceBuffer materialized = null;
    try {
      java.util.List<Tensor> traced = inputs.stream()
          .map(input -> new Tensor(fusing, fusing.leaf(input.buffer()))).toList();
      Tensor output = body.apply(traced);
      materialized = fusing.materialize(output.buffer());
      for (Tensor input : inputs) if (materialized == input.buffer()) return input;
      return new Tensor(real, materialized);
    } finally {
      fusing.releaseIntermediates(materialized);
    }
  }

  /** Captures a pure function once per shape/dtype/device/backend signature.
   * Runtime values remain inputs. Closed-over tensors must remain alive;
   * Java side effects and scalar constants are evaluated only during capture.
   */
  public static Compiled compile(UnaryOperator<Tensor> body) {
    return new Compiled(body);
  }

  public static final class Compiled implements UnaryOperator<Tensor> {
    private final UnaryOperator<Tensor> body;
    private final java.util.Map<Signature, ExecutionPlan> plans = new java.util.HashMap<>();

    private Compiled(UnaryOperator<Tensor> body) {
      this.body = java.util.Objects.requireNonNull(body);
    }

    @Override
    public synchronized Tensor apply(Tensor input) {
      Signature signature = new Signature(input.shape(), input.dtype(), input.device(), input.backend());
      ExecutionPlan plan = plans.get(signature);
      if (plan != null) return (Tensor) plan.execute(java.util.List.of(input));
      ExecutionPlan.Capture capture = ExecutionPlan.capture(xs -> body.apply(xs.get(0)), java.util.List.of(input));
      plans.put(signature, capture.plan());
      return (Tensor) capture.output();
    }

    public synchronized int cachedPlanCount() { return plans.size(); }
  }

  /** Captures a Tensor-leaf pytree and preserves the output tree, including value/gradient records. */
  public static <P, R> CompiledTree<P, R> compileTree(java.util.function.Function<P, R> body) {
    return new CompiledTree<>(body);
  }

  public static final class CompiledTree<P, R> implements java.util.function.Function<P, R> {
    private final java.util.function.Function<P, R> body;
    private final java.util.Map<TreeSignature, ExecutionPlan> plans = new java.util.HashMap<>();
    private CompiledTree(java.util.function.Function<P,R> body) { this.body=java.util.Objects.requireNonNull(body); }
    @SuppressWarnings("unchecked")
    public synchronized R apply(P input) {
      var flat=com.nablatensor.jax4j.tensor.tree.TreeUtil.flatten(input);
      java.util.List<Tensor> tensors=tensorLeaves(flat.leaves());
      if(tensors.isEmpty()) throw new IllegalArgumentException("Compiled trees require Tensor inputs");
      TreeSignature signature=new TreeSignature(flat.def(),tensors.stream().map(x->new Signature(x.shape(),x.dtype(),x.device(),x.backend())).toList());
      ExecutionPlan plan=plans.get(signature);
      if(plan!=null) return (R)plan.execute(tensors);
      ExecutionPlan.Capture capture=ExecutionPlan.capture(xs->body.apply((P)com.nablatensor.jax4j.tensor.tree.TreeUtil.unflatten(flat.def(),xs)),tensors);
      plans.put(signature,capture.plan());
      return (R)capture.output();
    }
    public synchronized int cachedPlanCount(){return plans.size();}
  }

  private record TreeSignature(com.nablatensor.jax4j.tensor.tree.TreeDef definition,java.util.List<Signature> leaves) {}

  static java.util.List<Tensor> tensorLeaves(java.util.List<Object> leaves) {
    return leaves.stream().map(x->{
      if(!(x instanceof Tensor tensor)) throw new IllegalArgumentException("Compiled trees require Tensor leaves");
      return tensor;
    }).toList();
  }

  static Object applyTree(java.util.function.Function<java.util.List<Tensor>,Object> body,
                          java.util.List<Tensor> inputs, ComputeBackend outputBackend) {
    ComputeBackend real=inputs.get(0).backend();
    for(Tensor input:inputs) if(input.backend()!=real || !input.device().equals(inputs.get(0).device()))
      throw new IllegalArgumentException("Compiled inputs must share backend and device");
    FusingBackend fusing=new FusingBackend(real);
    java.util.List<DeviceBuffer> keep=new java.util.ArrayList<>();
    boolean success=false;
    try {
      var traced=inputs.stream().map(x->new Tensor(fusing,fusing.leaf(x.buffer()))).toList();
      var output=com.nablatensor.jax4j.tensor.tree.TreeUtil.flatten(body.apply(traced));
      var tensors=tensorLeaves(output.leaves());
      java.util.List<Tensor> results=new java.util.ArrayList<>();
      var wrappers=new java.util.IdentityHashMap<DeviceBuffer,Tensor>();
      for(Tensor tensor:tensors) {
        DeviceBuffer buffer=fusing.materialize(tensor.buffer()); keep.add(buffer);
        results.add(wrappers.computeIfAbsent(buffer,b->new Tensor(outputBackend,b)));
      }
      success=true;
      return com.nablatensor.jax4j.tensor.tree.TreeUtil.unflatten(output.def(),results);
    } finally { fusing.releaseIntermediates(success?keep:java.util.List.of()); }
  }

  private record Signature(Shape shape, DType dtype, Device device, ComputeBackend backend) {}
}
