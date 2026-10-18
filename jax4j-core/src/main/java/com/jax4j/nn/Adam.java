package com.jax4j.nn;

import com.jax4j.tensor.Jit;
import com.jax4j.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;

/** Adam with device-resident moments and fused elementwise updates. */
public final class Adam implements AutoCloseable {
  private final float rate;
  private final List<Tensor> m = new ArrayList<>(), v = new ArrayList<>();
  private int step;

  public Adam(float rate) {
    if (!(rate > 0) || !Float.isFinite(rate))
      throw new IllegalArgumentException("Positive finite learning rate required");
    this.rate = rate;
  }

  /**
   * Returns new parameters. The caller owns parameters and gradients; this optimizer owns moments.
   */
  public synchronized List<Tensor> update(List<Tensor> parameters, List<Tensor> gradients) {
    if (parameters.stream().anyMatch(Tensor::isTracing))
      throw new IllegalStateException(
          "Stateful Adam cannot mutate inside a transformation; use FunctionalAdam with explicit"
              + " state");
    if (parameters.size() != gradients.size() || (step > 0 && parameters.size() != m.size()))
      throw new IllegalArgumentException("Parameter/gradient count changed");
    for (int i = 0; i < parameters.size(); i++) {
      Tensor p = parameters.get(i), g = gradients.get(i);
      if (!p.shape().equals(g.shape()) || p.dtype() != g.dtype() || !p.device().equals(g.device()))
        throw new IllegalArgumentException("Parameter/gradient shape or placement mismatch");
      if (step > 0
          && (!p.shape().equals(m.get(i).shape()) || !p.device().equals(m.get(i).device())))
        throw new IllegalArgumentException("Optimizer shape or placement changed");
    }
    List<Tensor> nextM = new ArrayList<>(), nextV = new ArrayList<>(), result = new ArrayList<>();
    int nextStep = Math.incrementExact(step);
    double bc1 = 1 - Math.pow(.9, nextStep), bc2 = 1 - Math.pow(.999, nextStep);
    try {
      for (int i = 0; i < parameters.size(); i++) {
        Tensor g = gradients.get(i), p = parameters.get(i);
        Tensor mm, vv;
        if (step == 0) {
          mm = Jit.apply(x -> x.mul(.1), g);
          vv = Jit.apply(x -> x.mul(x).mul(.001), g);
        } else {
          Tensor oldM = m.get(i), oldV = v.get(i);
          mm = Jit.apply(xs -> xs.get(0).mul(.1).add(xs.get(1).mul(.9)), List.of(g, oldM));
          vv =
              Jit.apply(
                  xs -> xs.get(0).mul(xs.get(0)).mul(.001).add(xs.get(1).mul(.999)),
                  List.of(g, oldV));
        }
        nextM.add(mm);
        nextV.add(vv);
        result.add(
            Jit.apply(
                xs ->
                    xs.get(0)
                        .sub(xs.get(1).div(bc1).div(xs.get(2).div(bc2).sqrt().add(1e-8)).mul(rate)),
                List.of(p, mm, vv)));
      }
    } catch (RuntimeException | Error failure) {
      nextM.forEach(Tensor::close);
      nextV.forEach(Tensor::close);
      result.forEach(Tensor::close);
      throw failure;
    }
    m.forEach(Tensor::close);
    v.forEach(Tensor::close);
    m.clear();
    m.addAll(nextM);
    v.clear();
    v.addAll(nextV);
    step = nextStep;
    return List.copyOf(result);
  }

  @Override
  public synchronized void close() {
    m.forEach(Tensor::close);
    v.forEach(Tensor::close);
    m.clear();
    v.clear();
    step = 0;
  }
}
