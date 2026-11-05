package com.nablatensor.jax4j.tensor;

/** Shared batch-2 elementwise semantics. Masks are F32 zero/one, not a new storage dtype. */
public final class Elementwise {
  private Elementwise() {}

  public static float binary(Op op, float x, float y) {
    return switch (op) {
      case LT -> x < y ? 1 : 0;
      case LE -> x <= y ? 1 : 0;
      case GT -> x > y ? 1 : 0;
      case GE -> x >= y ? 1 : 0;
      case EQ -> x == y ? 1 : 0;
      case NE -> x != y ? 1 : 0;
      case AND -> x != 0 && y != 0 ? 1 : 0;
      case OR -> x != 0 || y != 0 ? 1 : 0;
      case XOR -> (x != 0) != (y != 0) ? 1 : 0;
      case POW -> (float) Math.pow(x, y);
      case HYPOT -> (float) Math.hypot(x, y);
      default -> throw new IllegalArgumentException("not a batch-2 binary op: " + op);
    };
  }

  public static float unary(Op op, float x) {
    return switch (op) {
      case SIN -> (float) Math.sin(x);
      case COS -> (float) Math.cos(x);
      case NOT -> x == 0 ? 1 : 0;
      case ISFINITE -> Float.isFinite(x) ? 1 : 0;
      case ISNAN -> Float.isNaN(x) ? 1 : 0;
      default -> throw new IllegalArgumentException("not a batch-2 unary op: " + op);
    };
  }

  public static String binarySource(Op op, String left, String right) {
    String x = "(" + left + ")", y = "(" + right + ")";
    return switch (op) {
      case LT, LE, GT, GE, EQ, NE ->
          "("
              + x
              + switch (op) {
                case LT -> "<";
                case LE -> "<=";
                case GT -> ">";
                case GE -> ">=";
                case EQ -> "==";
                default -> "!=";
              }
              + y
              + " ? 1.0 : 0.0)";
      case AND -> "((" + x + "!=0.0 && " + y + "!=0.0) ? 1.0 : 0.0)";
      case OR -> "((" + x + "!=0.0 || " + y + "!=0.0) ? 1.0 : 0.0)";
      case XOR -> "(((" + x + "!=0.0) != (" + y + "!=0.0)) ? 1.0 : 0.0)";
      case POW -> powerSource(x, y);
      case HYPOT ->
          "((isinf("
              + x
              + ") || isinf("
              + y
              + ")) ? uintBitsToFloat(0x7f800000u) : ((isnan("
              + x
              + ") || isnan("
              + y
              + ")) ? uintBitsToFloat(0x7fc00000u) : (max(abs("
              + x
              + "),abs("
              + y
              + ")) == 0.0 ? 0.0 : max(abs("
              + x
              + "),abs("
              + y
              + "))*sqrt(1.0+pow(min(abs("
              + x
              + "),abs("
              + y
              + "))/max(abs("
              + x
              + "),abs("
              + y
              + ")),2.0)))))";
      default -> throw new IllegalArgumentException("not a batch-2 binary op: " + op);
    };
  }

  private static String powerSource(String x, String y) {
    String nan = "uintBitsToFloat(0x7fc00000u)", inf = "uintBitsToFloat(0x7f800000u)";
    String odd = "(floor(" + y + ")==" + y + " && mod(abs(" + y + "),2.0)==1.0)";
    String sign = "((floatBitsToInt(" + x + ")<0 && " + odd + ") ? -1.0 : 1.0)";
    String finite =
        "("
            + x
            + "<0.0 ? (floor("
            + y
            + ")=="
            + y
            + " ? "
            + sign
            + "*pow(-"
            + x
            + ","
            + y
            + ") : "
            + nan
            + ") : pow("
            + x
            + ","
            + y
            + "))";
    String magnitude =
        "(isinf("
            + x
            + ") ? "
            + sign
            + "*("
            + y
            + ">0.0 ? "
            + inf
            + " : 0.0) : ("
            + x
            + "==0.0 ? "
            + sign
            + "*("
            + y
            + ">0.0 ? 0.0 : "
            + inf
            + ") : "
            + finite
            + "))";
    return "("
        + y
        + "==0.0 ? 1.0 : ((isnan("
        + x
        + ") || isnan("
        + y
        + ")) ? "
        + nan
        + " : (isinf("
        + y
        + ") ? (abs("
        + x
        + ")==1.0 ? "
        + nan
        + " : ((abs("
        + x
        + ")>1.0)==("
        + y
        + ">0.0) ? "
        + inf
        + " : 0.0)) : "
        + magnitude
        + ")))";
  }

  public static String unarySource(Op op, String x) {
    return switch (op) {
      case SIN -> "sin(" + x + ")";
      case COS -> "cos(" + x + ")";
      case NOT -> "((" + x + ")==0.0 ? 1.0 : 0.0)";
      case ISFINITE -> "((!isnan(" + x + ") && !isinf(" + x + ")) ? 1.0 : 0.0)";
      case ISNAN -> "(isnan(" + x + ") ? 1.0 : 0.0)";
      default -> throw new IllegalArgumentException("not a batch-2 unary op: " + op);
    };
  }

  public static String cuda(String source) {
    return source
        .replace("floatBitsToInt(", "__float_as_int(")
        .replace("pow(", "powf(")
        .replace("mod(", "fmodf(")
        .replace("uintBitsToFloat(0x7f800000u)", "__int_as_float(0x7f800000)")
        .replace("uintBitsToFloat(0x7fc00000u)", "__int_as_float(0x7fc00000)");
  }
}
