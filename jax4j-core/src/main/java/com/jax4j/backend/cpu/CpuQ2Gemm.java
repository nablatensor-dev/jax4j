package com.jax4j.backend.cpu;

/**
 * Fused dequantize-and-multiply kernel for the ggml Q2_0 ternary format.
 * Duplicated from {@code com.jax4j.llm.quant.Q2Gemm} (tensorjit-backend-cpu
 * cannot depend on tensorjit-llm); weights stay quantized in memory, dequantized
 * on the fly inside the dot product.
 */
final class CpuQ2Gemm {

  private static final int BLOCK = 128;
  private static final int BLOCK_BYTES = 34;

  private CpuQ2Gemm() {
  }

  static void gather(float[] out, byte[] weights, int token, int rows, int columns) {
    validate(rows, columns);
    int rowBytes = columns / BLOCK * BLOCK_BYTES;
    int rowOffset = token * rowBytes;
    for (int index = 0; index < columns; index++) {
      out[index] = packedValue(weights, rowOffset, index);
    }
  }

  static void multiply(float[] out, float[] input, byte[] weights, int rows, int columns) {
    validate(rows, columns);
    int blocks = columns / BLOCK;
    for (int row = 0; row < rows; row++) {
      float sum = 0.0f;
      int rowOffset = row * blocks * BLOCK_BYTES;
      for (int block = 0; block < blocks; block++) {
        int blockOffset = rowOffset + block * BLOCK_BYTES;
        float scale = halfToFloat((weights[blockOffset] & 0xff) | ((weights[blockOffset + 1] & 0xff) << 8));
        int inputBase = block * BLOCK;
        for (int packedIndex = 0; packedIndex < 32; packedIndex++) {
          int packed = weights[blockOffset + 2 + packedIndex] & 0xff;
          int index = inputBase + (packedIndex << 2);
          sum += scale * ((packed & 3) - 1) * input[index];
          sum += scale * (((packed >>> 2) & 3) - 1) * input[index + 1];
          sum += scale * (((packed >>> 4) & 3) - 1) * input[index + 2];
          sum += scale * (((packed >>> 6) & 3) - 1) * input[index + 3];
        }
      }
      out[row] = sum;
    }
  }

  private static float packedValue(byte[] weights, int rowOffset, int index) {
    int block = index / BLOCK;
    int within = index % BLOCK;
    int offset = rowOffset + block * BLOCK_BYTES;
    float scale = halfToFloat((weights[offset] & 0xff) | ((weights[offset + 1] & 0xff) << 8));
    int packed = weights[offset + 2 + within / 4] & 0xff;
    return scale * (((packed >>> ((within & 3) * 2)) & 3) - 1);
  }

  private static float halfToFloat(int bits) {
    int sign = (bits & 0x8000) << 16;
    int exponent = (bits >>> 10) & 0x1f;
    int fraction = bits & 0x3ff;
    if (exponent == 0) {
      if (fraction == 0) {
        return Float.intBitsToFloat(sign);
      }
      exponent = 1;
      while ((fraction & 0x400) == 0) {
        fraction <<= 1;
        exponent--;
      }
      return Float.intBitsToFloat(sign | ((exponent + 112) << 23) | ((fraction & 0x3ff) << 13));
    }
    if (exponent == 31) {
      return Float.intBitsToFloat(sign | 0x7f800000 | (fraction << 13));
    }
    return Float.intBitsToFloat(sign | ((exponent + 112) << 23) | (fraction << 13));
  }

  private static void validate(int rows, int columns) {
    if (rows <= 0 || columns <= 0 || columns % BLOCK != 0) {
      throw new IllegalArgumentException("invalid Q2_0 GEMM dimensions: rows=" + rows + " columns=" + columns);
    }
  }
}
