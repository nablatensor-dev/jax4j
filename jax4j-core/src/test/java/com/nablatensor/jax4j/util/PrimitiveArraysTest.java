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

import static com.nablatensor.jax4j.util.PrimitiveArrays.*;
import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class PrimitiveArraysTest {
  @Test
  void buildsArraysOfAllEightPrimitiveTypes() {
    assertArrayEquals(new float[] {1.23f, 4.56f}, floats(1.23, 4.56));
    assertArrayEquals(new double[] {1.23, 4.56}, doubles(1.23, 4.56));
    assertArrayEquals(new int[] {1, -2}, ints(1, -2));
    assertArrayEquals(new long[] {1, 3_000_000_000L}, longs(1, 3_000_000_000L));
    assertArrayEquals(new short[] {Short.MIN_VALUE, Short.MAX_VALUE}, shorts(-32768, 32767));
    assertArrayEquals(new byte[] {Byte.MIN_VALUE, Byte.MAX_VALUE}, bytes(-128, 127));
    assertArrayEquals(new char[] {'a', 'λ'}, chars('a', 'λ'));
    assertArrayEquals(new boolean[] {true, false}, booleans(true, false));
  }

  @Test
  void acceptsEmptyArgumentLists() {
    assertArrayEquals(new float[0], floats());
    assertArrayEquals(new double[0], doubles());
    assertArrayEquals(new int[0], ints());
    assertArrayEquals(new long[0], longs());
    assertArrayEquals(new short[0], shorts());
    assertArrayEquals(new byte[0], bytes());
    assertArrayEquals(new char[0], chars());
    assertArrayEquals(new boolean[0], booleans());
  }

  @Test
  void appliesJavaFloatConversion() {
    assertArrayEquals(
        new float[] {1.0f, -0.0f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY},
        floats(1.00000001, -0.0, Double.NaN, Double.MAX_VALUE, Double.NEGATIVE_INFINITY));
  }

  @Test
  void reportsValuesOutsideSignedIntegerRanges() {
    assertEquals("Value 128 at index 1 is out of byte range",
        assertThrows(IllegalArgumentException.class, () -> bytes(0, 128)).getMessage());
    assertEquals("Value -129 at index 0 is out of byte range",
        assertThrows(IllegalArgumentException.class, () -> bytes(-129)).getMessage());
    assertEquals("Value 32768 at index 1 is out of short range",
        assertThrows(IllegalArgumentException.class, () -> shorts(0, 32768)).getMessage());
    assertEquals("Value -32769 at index 0 is out of short range",
        assertThrows(IllegalArgumentException.class, () -> shorts(-32769)).getMessage());
  }
}
