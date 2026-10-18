package com.jax4j.autodiff;

import com.jax4j.tensor.ConvSpec;
import com.jax4j.tensor.Device;
import com.jax4j.tensor.Tensors;
import com.jax4j.tensor.Shape;
import com.jax4j.tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

/**
 * A reverse-mode autodiff tape node wrapping a {@link Tensor}. Every fluent op
 * (add/mul/matmul/...) both computes the forward {@link Tensor} value and
 * records a local backward rule; {@link #backward()} walks the recorded graph
 * in reverse topological order, accumulating each node's gradient into its
 * parents. {@link AutoDiff} brackets a recording session so the intermediate
 * (non-leaf) nodes created along the way can be closed deterministically once
 * gradients have been read out, instead of relying on GC/Cleaner timing.
 */
public final class Var {

  private final Tensor value;
  private final List<Var> parents;
  private final BackwardRule backwardRule;
  private Tensor grad;

  @FunctionalInterface
  private interface BackwardRule {
    void run(Tensor upstream);
  }

  private Var(Tensor value, List<Var> parents, BackwardRule backwardRule, boolean tracked) {
    this.value = value;
    this.parents = parents;
    this.backwardRule = backwardRule;
    if (tracked) {
      Tape.record(this);
    }
  }

  /** Wraps a tensor as an untracked leaf: a trainable parameter or a constant. Never auto-closed. */
  public static Var leaf(Tensor value) {
    return new Var(value, List.of(), null, false);
  }

  private static Var traced(Tensor value, List<Var> parents, BackwardRule rule) {
    return new Var(value, parents, rule, true);
  }

  public Tensor value() {
    return value;
  }

  /** Accumulated upstream gradient after {@link #backward()}; {@code null} if never reached. */
  public Tensor grad() {
    return grad;
  }

  private void accumulate(Tensor upstream) {
    if (grad == null) {
      grad = upstream;
    } else {
      Tensor merged = grad.add(upstream);
      grad.close();
      upstream.close();
      grad = merged;
    }
  }

  // ---- ops ------------------------------------------------------------------

  public Var add(Var other) {
    Tensor out = this.value.add(other.value);
    return traced(out, List.of(this, other), upstream -> {
      this.accumulate(reduceGradTo(dup(upstream), this.value.shape()));
      other.accumulate(reduceGradTo(upstream, other.value.shape()));
    });
  }

  public Var sub(Var other) {
    Tensor out = this.value.sub(other.value);
    return traced(out, List.of(this, other), upstream -> {
      this.accumulate(reduceGradTo(dup(upstream), this.value.shape()));
      Tensor negUpstream = upstream.neg();
      other.accumulate(reduceGradTo(negUpstream, other.value.shape()));
    });
  }

  public Var mul(Var other) {
    Tensor out = this.value.mul(other.value);
    return traced(out, List.of(this, other), upstream -> {
      Tensor dThisRaw = upstream.mul(other.value);
      this.accumulate(reduceGradTo(dThisRaw, this.value.shape()));
      Tensor dOtherRaw = upstream.mul(this.value);
      other.accumulate(reduceGradTo(dOtherRaw, other.value.shape()));
    });
  }

  public Var div(Var other) {
    Tensor out = this.value.div(other.value);
    return traced(out, List.of(this, other), upstream -> {
      Tensor dThisRaw = upstream.div(other.value);
      this.accumulate(reduceGradTo(dThisRaw, this.value.shape()));

      Tensor numerator = upstream.mul(this.value);
      Tensor denominator = other.value.mul(other.value);
      Tensor quotient = numerator.div(denominator);
      numerator.close();
      denominator.close();
      other.accumulate(reduceGradTo(quotient.neg(), other.value.shape()));
      quotient.close();
    });
  }

  /**
   * Returns an independent copy of {@code t} (fresh backing buffer). Needed
   * whenever the same upstream gradient tensor is about to be handed to two
   * different accumulate() calls (e.g. both operands of add/sub) - without
   * this, both Vars' {@code grad} could end up aliasing the exact same
   * buffer, and closing/merging one would silently corrupt the other
   * (harmless-looking on CPU's no-op close, but a real use-after-free on
   * CUDA's Cleaner-backed release).
   */
  private static Tensor dup(Tensor t) {
    return t.mul(1.0);
  }

  public Var matmul(Var other) {
    Tensor out = this.value.matmul(other.value);
    return traced(out, List.of(this, other), upstream -> {
      Tensor otherT = other.value.transpose();
      Tensor dThis = upstream.matmul(otherT);
      otherT.close();
      this.accumulate(dThis);

      Tensor thisT = this.value.transpose();
      Tensor dOther = thisT.matmul(upstream);
      thisT.close();
      other.accumulate(dOther);
    });
  }

  public Var transpose() {
    Tensor out = this.value.transpose();
    return traced(out, List.of(this), upstream -> this.accumulate(upstream.transpose()));
  }

  public Var reshape(int... dims) {
    Shape inputShape = this.value.shape();
    Tensor out = this.value.reshape(dims);
    return traced(
        out,
        List.of(this),
        upstream -> this.accumulate(upstream.reshape(inputShape.dims())));
  }

  /** Rectified linear unit; the backward pass routes gradient only where the input was positive. */
  public Var relu() {
    Tensor out = this.value.relu();
    return traced(out, List.of(this), upstream -> this.accumulate(upstream.reluBackward(this.value)));
  }

  /** 2-D convolution against {@code weights}; see {@link ConvSpec} for the layout. */
  public Var conv2d(Var weights, ConvSpec spec) {
    Tensor out = this.value.conv2d(weights.value, spec);
    return traced(out, List.of(this, weights), upstream -> {
      this.accumulate(upstream.conv2dGradInput(weights.value, spec));
      weights.accumulate(this.value.conv2dGradWeight(upstream, spec));
    });
  }

  public Var exp() {
    Tensor out = this.value.exp();
    return traced(out, List.of(this), upstream -> this.accumulate(upstream.mul(out)));
  }

  public Var log() {
    Tensor out = this.value.log();
    return traced(out, List.of(this), upstream -> this.accumulate(upstream.div(this.value)));
  }

  public Var sqrt() {
    Tensor out = this.value.sqrt();
    return traced(out, List.of(this), upstream -> {
      Tensor denominator = out.mul(2.0);
      Tensor dIn = upstream.div(denominator);
      denominator.close();
      this.accumulate(dIn);
    });
  }

  public Var tanh() {
    Tensor out = this.value.tanh();
    return traced(out, List.of(this), upstream -> {
      Tensor squared = out.mul(out);
      Tensor negSquared = squared.neg();
      squared.close();
      Tensor localGrad = negSquared.add(1.0);
      negSquared.close();
      Tensor dIn = upstream.mul(localGrad);
      localGrad.close();
      this.accumulate(dIn);
    });
  }

  public Var abs() {
    Tensor out = this.value.abs();
    return traced(out, List.of(this), upstream -> {
      Tensor sign = this.value.sign();
      Tensor dIn = upstream.mul(sign);
      sign.close();
      this.accumulate(dIn);
    });
  }

  public Var neg() {
    Tensor out = this.value.neg();
    return traced(out, List.of(this), upstream -> this.accumulate(upstream.neg()));
  }

  public Var sigmoid() {
    Tensor out = this.value.sigmoid();
    return traced(out, List.of(this), upstream -> {
      Tensor negOut = out.neg();
      Tensor oneMinusOut = negOut.add(1.0);
      negOut.close();
      Tensor localGrad = out.mul(oneMinusOut);
      oneMinusOut.close();
      Tensor dIn = upstream.mul(localGrad);
      localGrad.close();
      this.accumulate(dIn);
    });
  }

  /** Scales by a plain (non-differentiable) double, e.g. a learning rate or {@code 1/N}. */
  public Var scaled(double factor) {
    Tensor out = this.value.mul(factor);
    return traced(out, List.of(this), upstream -> this.accumulate(upstream.mul(factor)));
  }

  /** Sums every element, returning a single-element (shape {@code (1)}) node. */
  public Var sum() {
    Shape inShape = this.value.shape();
    Tensor out = this.value.sum();
    return traced(out, List.of(this), upstream -> this.accumulate(upstream.broadcastTo(inShape)));
  }

  /** Sums along one axis. The backward pass broadcasts the reduced gradient to the input shape. */
  public Var sum(int axis) {
    return sum(axis, false);
  }

  /** Sums along one axis, optionally retaining a size-one reduced dimension. */
  public Var sum(int axis, boolean keepDims) {
    Shape inputShape = this.value.shape();
    int normalizedAxis = normalizeAxis(axis, inputShape);
    Tensor out = this.value.sum(normalizedAxis, keepDims);
    return traced(out, List.of(this), upstream -> {
      int[] expandedDims = inputShape.dims();
      expandedDims[normalizedAxis] = 1;
      Tensor expanded = upstream.reshape(expandedDims);
      Tensor broadcast = expanded.broadcastTo(inputShape);
      if (broadcast != expanded) {
        expanded.close();
      }
      this.accumulate(broadcast);
    });
  }

  public Var max(int axis) {
    return max(axis, false);
  }

  /**
   * Maximum along one axis. Ties route the gradient to the first maximum,
   * matching {@link Tensor#argmax(int)}.
   */
  public Var max(int axis, boolean keepDims) {
    Shape inputShape = this.value.shape();
    int normalizedAxis = normalizeAxis(axis, inputShape);
    Tensor out = this.value.max(normalizedAxis, keepDims);
    return traced(out, List.of(this), upstream ->
        this.accumulate(this.value.maxAxisBackward(upstream, normalizedAxis)));
  }

  /** Mean over every element, returning a single-element (shape {@code (1)}) node. */
  public Var mean() {
    return sum().scaled(1.0 / this.value.shape().size());
  }

  // ---- backward ---------------------------------------------------------

  /** Seeds this (scalar-shaped) node's gradient with 1 and backprops through the whole tape. */
  public void backward() {
    List<Var> topo = new ArrayList<>();
    Set<Var> visited = Collections.newSetFromMap(new IdentityHashMap<>());
    buildTopo(this, visited, topo);

    if (value.shape().size() != 1) throw new IllegalArgumentException("backward requires a scalar loss");
    this.grad = this.value.fullLike(1f);

    for (int i = topo.size() - 1; i >= 0; i--) {
      Var v = topo.get(i);
      if (v.backwardRule != null && v.grad != null) {
        v.backwardRule.run(v.grad);
      }
    }
  }

  private static void buildTopo(Var v, Set<Var> visited, List<Var> topo) {
    if (!visited.add(v)) {
      return;
    }
    for (Var parent : v.parents) {
      buildTopo(parent, visited, topo);
    }
    topo.add(v);
  }

  /**
   * Reduces a gradient down to {@code target}'s shape when the forward op
   * broadcast a smaller operand up to a bigger one.
   * Consumes (and closes, if reduced) {@code grad}.
   */
  private static Tensor reduceGradTo(Tensor grad, Shape target) {
    if (grad.shape().equals(target)) {
      return grad;
    }
    Shape source = grad.shape();
    if (target.rank() > source.rank() || !Shape.broadcast(target, source).equals(source)) {
      grad.close();
      throw new UnsupportedOperationException(
          "Var: illegal broadcast-gradient reduction " + source + " -> " + target);
    }

    Tensor reduced = grad;
    while (reduced.shape().rank() > target.rank()) {
      Tensor next = reduced.sum(0);
      reduced.close();
      reduced = next;
    }
    for (int axis = 0; axis < target.rank(); axis++) {
      if (target.dim(axis) == 1 && reduced.shape().dim(axis) > 1) {
        Tensor next = reduced.sum(axis, true);
        reduced.close();
        reduced = next;
      }
    }
    if (!reduced.shape().equals(target)) {
      Tensor reshaped = reduced.reshape(target.dims());
      reduced.close();
      reduced = reshaped;
    }
    return reduced;
  }

  private static int normalizeAxis(int axis, Shape shape) {
    int normalized = axis < 0 ? axis + shape.rank() : axis;
    if (normalized < 0 || normalized >= shape.rank()) {
      throw new IllegalArgumentException("axis " + axis + " is out of bounds for shape " + shape);
    }
    return normalized;
  }
}
