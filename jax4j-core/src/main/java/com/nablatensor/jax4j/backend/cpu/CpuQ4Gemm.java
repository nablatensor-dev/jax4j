package com.nablatensor.jax4j.backend.cpu;

/**
 * Fused dequantize-and-multiply kernel for the ggml Q4_0 format (32
 * weights / 18 bytes per block, {@code weight = d*(nibble-8)}, low nibble is
 * element i, high nibble is element i+16). Phase 8 "insurance" format --
 * validates tensorjit-llm isn't permanently welded to the prism-fork Q2_0 type.
 */
final class CpuQ4Gemm {

  private static final int BLOCK = 32;
  private static final int BLOCK_BYTES = 18;

  private CpuQ4Gemm() {
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
        for (int packedIndex = 0; packedIndex < 16; packedIndex++) {
          int packed = weights[blockOffset + 2 + packedIndex] & 0xff;
          int low = packed & 0x0f;
          int high = (packed >>> 4) & 0x0f;
          sum += scale * (low - 8) * input[inputBase + packedIndex];
          sum += scale * (high - 8) * input[inputBase + packedIndex + 16];
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
    int packed = weights[offset + 2 + (within % 16)] & 0xff;
    int nibble = within < 16 ? (packed & 0x0f) : ((packed >>> 4) & 0x0f);
    return scale * (nibble - 8);
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
      throw new IllegalArgumentException("invalid Q4_0 GEMM dimensions: rows=" + rows + " columns=" + columns);
    }
  }
}
