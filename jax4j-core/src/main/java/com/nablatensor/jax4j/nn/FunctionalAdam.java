package com.nablatensor.jax4j.nn;

import com.nablatensor.jax4j.tensor.*;
import com.nablatensor.jax4j.tensor.tree.TreeUtil;
import java.util.*;

/**
 * Pure Adam updates: moments and the device step counter are explicit pytree inputs and outputs.
 */
public final class FunctionalAdam {
  public record State<P>(P firstMoment, P secondMoment, Tensor step) {}

  public record Input<P>(P parameters, P gradients, State<P> state) {}

  public record Update<P>(P parameters, State<P> state) {}

  private final double rate, beta1, beta2, epsilon;
  private final Jit.CompiledTree<Input<Object>, Update<Object>> compiled;

  public FunctionalAdam(double rate) {
    this(rate, .9, .999, 1e-8);
  }

  public FunctionalAdam(double rate, double beta1, double beta2, double epsilon) {
    if (!(rate > 0)
        || !Double.isFinite(rate)
        || beta1 < 0
        || beta1 >= 1
        || beta2 < 0
        || beta2 >= 1
        || !Double.isFinite(beta1)
        || !Double.isFinite(beta2)
        || !(epsilon > 0)
        || !Double.isFinite(epsilon))
      throw new IllegalArgumentException("invalid Adam hyperparameters");
    this.rate = rate;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
    compiled = Jit.compileTree(this::step);
  }

  @SuppressWarnings("unchecked")
  public <P> State<P> initialize(P parameters) {
    var flat = TreeUtil.flatten(parameters);
    List<Tensor> ps = leaves(flat.leaves());
    if (ps.isEmpty()) throw new IllegalArgumentException("Adam requires parameters");
    List<Tensor> m = new ArrayList<>(), v = new ArrayList<>();
    for (Tensor p : ps) {
      m.add(p.fullLike(0));
      v.add(p.fullLike(0));
    }
    Tensor step;
    try (Tensor zeros = ps.getFirst().fullLike(0)) {
      step = zeros.sum();
    }
    return new State<>(
        (P) TreeUtil.unflatten(flat.def(), m), (P) TreeUtil.unflatten(flat.def(), v), step);
  }

  @SuppressWarnings("unchecked")
  public <P> Update<P> update(P parameters, P gradients, State<P> state) {
    return (Update<P>)
        (Object)
            compiled.apply(
                new Input<>(
                    (Object) parameters, (Object) gradients, (State<Object>) (Object) state));
  }

  private Update<Object> step(Input<Object> input) {
    var flat = TreeUtil.flatten(input.parameters());
    var grads = TreeUtil.flatten(input.gradients());
    var fm = TreeUtil.flatten(input.state().firstMoment());
    var sm = TreeUtil.flatten(input.state().secondMoment());
    if (!flat.def().equals(grads.def())
        || !flat.def().equals(fm.def())
        || !flat.def().equals(sm.def()))
      throw new IllegalArgumentException("Adam tree structure changed");
    List<Tensor> ps = leaves(flat.leaves()),
        gs = leaves(grads.leaves()),
        ms = leaves(fm.leaves()),
        vs = leaves(sm.leaves());
    Tensor nextStep = input.state().step().add(1);
    Tensor bc1 =
        beta1 == 0 ? nextStep.fullLike(1) : nextStep.mul(Math.log(beta1)).exp().neg().add(1);
    Tensor bc2 =
        beta2 == 0 ? nextStep.fullLike(1) : nextStep.mul(Math.log(beta2)).exp().neg().add(1);
    List<Tensor> nextM = new ArrayList<>(), nextV = new ArrayList<>(), nextP = new ArrayList<>();
    for (int i = 0; i < ps.size(); i++) {
      Tensor p = ps.get(i), g = gs.get(i), m = ms.get(i), v = vs.get(i);
      if (!p.shape().equals(g.shape())
          || !p.shape().equals(m.shape())
          || !p.shape().equals(v.shape()))
        throw new IllegalArgumentException("Adam parameter/gradient/state shapes differ");
      Tensor mm = m.mul(beta1).add(g.mul(1 - beta1)),
          vv = v.mul(beta2).add(g.mul(g).mul(1 - beta2));
      nextM.add(mm);
      nextV.add(vv);
      nextP.add(p.sub(mm.div(bc1).div(vv.div(bc2).sqrt().add(epsilon)).mul(rate)));
    }
    return new Update<>(
        TreeUtil.unflatten(flat.def(), nextP),
        new State<>(
            TreeUtil.unflatten(flat.def(), nextM),
            TreeUtil.unflatten(flat.def(), nextV),
            nextStep));
  }

  private static List<Tensor> leaves(List<Object> values) {
    return values.stream()
        .map(
            v -> {
              if (!(v instanceof Tensor t))
                throw new IllegalArgumentException("Adam requires Tensor leaves");
              return t;
            })
        .toList();
  }
}
