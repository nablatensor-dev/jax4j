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

/**
 * The recording engine and its backend SPI.
 *
 * <p><b>API</b> — {@link com.nablatensor.jax4j.engine.Nabla} (entry point),
 * {@link com.nablatensor.jax4j.engine.SDouble}, {@link com.nablatensor.jax4j.engine.AadRecorder},
 * {@link com.nablatensor.jax4j.engine.AadTape}, {@link com.nablatensor.jax4j.engine.AadResult},
 * {@link com.nablatensor.jax4j.engine.AadOptions}, {@link com.nablatensor.jax4j.engine.JitOptimizations}.
 *
 * <p><b>SPI</b> — implement {@link com.nablatensor.jax4j.engine.AadEngine} and
 * {@link com.nablatensor.jax4j.engine.AadExecutable} to add a backend; discovery is by
 * {@link java.util.ServiceLoader}. {@link com.nablatensor.jax4j.engine.AadEngines}
 * selects among them.
 *
 * <p>Types annotated {@link com.nablatensor.jax4j.annotation.Internal} (e.g.
 * {@code AbstractAadExecutable}, {@code CudaAadCodegen}, {@code AadCheckpointPlan})
 * are public only for cross-module reach and may change in any release. See
 * {@code docs/api-stability.md}.
 */
package com.nablatensor.jax4j.engine;
