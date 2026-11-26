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
package com.nablatensor.jax4j.util;

/** Factories for the eight primitive array types. All accept an empty argument list. */
public final class PrimitiveArrays {
  private PrimitiveArrays() {}

  /** Converts doubles to floats using Java's narrowing conversion. */
  public static float[] floats(double... values) {
    float[] result = new float[values.length];
    for (int i = 0; i < values.length; i++) {
      result[i] = (float) values[i];
    }
    return result;
  }

  /** Returns the varargs array. An explicitly supplied array is shared. */
  public static double[] doubles(double... values) {
    return values;
  }

  /** Returns the varargs array. An explicitly supplied array is shared. */
  public static int[] ints(int... values) {
    return values;
  }

  /** Returns the varargs array. An explicitly supplied array is shared. */
  public static long[] longs(long... values) {
    return values;
  }

  /**
   * Converts integers to shorts.
   *
   * @throws IllegalArgumentException if an element is outside the signed short range
   */
  public static short[] shorts(int... values) {
    short[] result = new short[values.length];
    for (int i = 0; i < values.length; i++) {
      int value = values[i];
      if (value < Short.MIN_VALUE || value > Short.MAX_VALUE) {
        throw new IllegalArgumentException("Value " + value + " at index " + i + " is out of short range");
      }
      result[i] = (short) value;
    }
    return result;
  }

  /**
   * Converts integers to bytes.
   *
   * @throws IllegalArgumentException if an element is outside the signed byte range
   */
  public static byte[] bytes(int... values) {
    byte[] result = new byte[values.length];
    for (int i = 0; i < values.length; i++) {
      int value = values[i];
      if (value < Byte.MIN_VALUE || value > Byte.MAX_VALUE) {
        throw new IllegalArgumentException("Value " + value + " at index " + i + " is out of byte range");
      }
      result[i] = (byte) value;
    }
    return result;
  }

  /** Returns the varargs array. An explicitly supplied array is shared. */
  public static char[] chars(char... values) {
    return values;
  }

  /** Returns the varargs array. An explicitly supplied array is shared. */
  public static boolean[] booleans(boolean... values) {
    return values;
  }
}
