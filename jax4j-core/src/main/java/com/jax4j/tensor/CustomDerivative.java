package com.jax4j.tensor;

import com.jax4j.tensor.spi.*;
import java.util.Objects;
import java.util.function.*;

/** A pure tensor primitive with an explicit derivative contract. Callbacks remain transformable. */
public final class CustomDerivative implements UnaryOperator<Tensor> {
  final UnaryOperator<Tensor> forward;
  final BiFunction<Tensor, Tensor, Tensor> jvp, vjp;

  CustomDerivative(
      UnaryOperator<Tensor> forward,
      BiFunction<Tensor, Tensor, Tensor> jvp,
      BiFunction<Tensor, Tensor, Tensor> vjp) {
    this.forward = Objects.requireNonNull(forward);
    this.jvp = jvp;
    this.vjp = vjp;
  }

  public static CustomDerivative jvp(
      UnaryOperator<Tensor> forward, BiFunction<Tensor, Tensor, Tensor> rule) {
    return new CustomDerivative(forward, Objects.requireNonNull(rule), null);
  }

  public static CustomDerivative vjp(
      UnaryOperator<Tensor> forward, BiFunction<Tensor, Tensor, Tensor> rule) {
    return new CustomDerivative(forward, null, Objects.requireNonNull(rule));
  }

  /**
   * Recompute the pure forward function during the pullback instead of retaining its internal tape.
   */
  public static CustomDerivative checkpoint(UnaryOperator<Tensor> forward) {
    return new CustomDerivative(
        x -> Jit.apply(forward, x),
        (x, t) -> {
          var result = Transforms.jvp(forward, x, t);
          if (result.value().buffer() != x.buffer()) result.value().close();
          return result.tangent();
        },
        (x, g) -> {
          var result = Transforms.vjp(forward, x, g);
          if (result.value() != x) result.value().close();
          return result.cotangents();
        });
  }

  public Tensor apply(Tensor input) {
    return new Tensor(input.backend(), input.backend().custom(input.buffer(), this));
  }

  /** Backend entry point; applications use apply. */
  public DeviceBuffer evaluate(ComputeBackend backend, DeviceBuffer input) {
    return forward.apply(new Tensor(backend, input)).buffer();
  }
}
