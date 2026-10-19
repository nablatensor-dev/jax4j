package com.nablatensor.jax4j.tensor;

/** ggml block-quantization formats. {@code id} is the on-disk GGUF type id. */
public enum QuantType {
  F32(0, 1, 4),
  F16(1, 1, 2),
  Q4_0(2, 32, 18),
  Q8_0(8, 32, 34),
  Q4_K(12, 256, 144),
  Q6_K(14, 256, 210),
  BF16(30, 1, 2),
  Q1_0(41, 256, 34), // prism fork, non-upstream
  Q2_0(42, 128, 34); // prism fork, non-upstream -- the pilot format

  private final int id;
  private final int blockElements;
  private final int blockBytes;

  QuantType(int id, int blockElements, int blockBytes) {
    this.id = id;
    this.blockElements = blockElements;
    this.blockBytes = blockBytes;
  }

  public int id() {
    return id;
  }

  public int blockElements() {
    return blockElements;
  }

  public int blockBytes() {
    return blockBytes;
  }

  public static QuantType byId(int id) {
    for (QuantType type : values()) {
      if (type.id == id) {
        return type;
      }
    }
    throw new IllegalArgumentException("unknown ggml quant type id: " + id);
  }

  /** Byte size of a tensor with {@code elements} logical elements of this type. */
  public long byteSizeFor(long elements) {
    if (elements % blockElements != 0) {
      throw new IllegalArgumentException(
          this + " tensor is not block-aligned: " + elements + " elements, block=" + blockElements);
    }
    return Math.multiplyExact(elements / blockElements, (long) blockBytes);
  }
}
