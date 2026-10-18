package com.jax4j.autodiff;

import java.util.ArrayList;
import java.util.List;

/**
 * Records every non-leaf {@link Var} created while a recording session is
 * active (single-threaded, matching the rest of tensorjit's Phase-0 assumption
 * that a notebook drives the backend from one thread). {@link AutoDiff}
 * starts/stops a session around each {@code grad}/{@code valueAndGrad} call
 * so intermediates can be closed deterministically afterwards, without the
 * caller having to track them by hand.
 */
final class Tape {

  private static final ThreadLocal<List<Var>> ACTIVE = new ThreadLocal<>();

  private Tape() {
  }

  static void record(Var v) {
    List<Var> active = ACTIVE.get();
    if (active != null) {
      active.add(v);
    }
  }

  static List<Var> begin() {
    if (ACTIVE.get() != null) throw new UnsupportedOperationException("Nested reverse-mode tapes require higher-order derivative rules");
    List<Var> trace = new ArrayList<>();
    ACTIVE.set(trace);
    return trace;
  }

  static void end() {
    ACTIVE.remove();
  }
}
