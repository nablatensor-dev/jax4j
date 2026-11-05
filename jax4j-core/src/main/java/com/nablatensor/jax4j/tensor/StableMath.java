package com.nablatensor.jax4j.tensor;

/** Stable GLSL expressions shared by eager, fused and structured-region GPU lowering. */
public final class StableMath {
  private StableMath() {}

  /** Horner series near zero avoids the loss of x in 1+x on F32 devices. */
  public static String log1p(String argument) {
    String x = "(" + argument + ")";
    String polynomial = Double.toString(-1.0 / 12);
    for (int n = 11; n >= 2; n--)
      polynomial = "(" + ((n % 2 == 0 ? -1.0 : 1.0) / n) + " + " + x + " * " + polynomial + ")";
    polynomial = "(" + x + " + " + x + " * " + x + " * " + polynomial + ")";
    return "(abs(" + x + ") < 0.125 ? " + polynomial + " : log(1.0 + " + x + "))";
  }

  public static String expm1(String argument) {
    String x = "(" + argument + ")";
    double factorial = 1;
    for (int n = 2; n <= 10; n++) factorial *= n;
    String polynomial = Double.toString(1 / factorial);
    for (int n = 9; n >= 2; n--) {
      factorial /= n + 1;
      polynomial = "(" + (1 / factorial) + " + " + x + " * " + polynomial + ")";
    }
    polynomial = "(" + x + " + " + x + " * " + x + " * " + polynomial + ")";
    return "(" + x + " == 0.0 ? " + x + " : (abs(" + x + ") < 0.5 ? " + polynomial + " : (exp(" + x + ") - 1.0)))";
  }

  public static String softplus(String argument) {
    return "(max((" + argument + "), 0.0) + " + log1p("exp(-abs(" + argument + "))") + ")";
  }
}
