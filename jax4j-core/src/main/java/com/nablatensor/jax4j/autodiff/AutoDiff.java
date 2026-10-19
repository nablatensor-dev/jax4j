package com.nablatensor.jax4j.autodiff;

import com.nablatensor.jax4j.tensor.Tensor;
import com.nablatensor.jax4j.tensor.tree.TreeDef;
import com.nablatensor.jax4j.tensor.tree.TreeUtil;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/**
 * Reverse-mode automatic differentiation over flat or structured parameters,
 * mirroring how the Python notebooks call
 * {@code jax.value_and_grad(loss)(params, x, y)}. Other arguments (batch data,
 * constants) are simply closed over by {@code lossFn}.
 *
 * <p>Every intermediate {@link Var} built while {@code lossFn} runs is closed
 * automatically once gradients have been read out, so callers only need to
 * manage the lifetime of the returned value/gradient tensors themselves.
 */
public final class AutoDiff {

  private AutoDiff() {
  }

  /** The scalar loss value and one gradient tensor per input parameter, in the same order. */
  public record ValueAndGrad(Tensor value, List<Tensor> grads) {
  }

  /** The scalar loss value and gradients rebuilt into the parameter tree's shape. */
  public record ValueAndGradTree<T>(Tensor value, T grads) {
  }

  /**
   * Computes a value and gradient for a structured parameter tree. Generic
   * records allow the parameter and variable trees to share a runtime shape
   * while using different leaf types, for example {@code Net<Tensor>} and
   * {@code Net<Var>}.
   */
  public static <P, V> ValueAndGradTree<P> valueAndGrad(
      Function<? super V, Var> lossFn, P paramTree) {
    TreeDef def = variableTreeDef(TreeUtil.structure(paramTree));
    return valueAndGrad(
        lossFn,
        paramTree,
        vars -> variableTree(def, vars));
  }

  /**
   * Computes a value and gradient for a structured parameter tree.
   *
   * <p>The factory receives parameter variables in deterministic pytree order
   * and returns the tree passed to {@code lossFn}. This keeps generic records
   * type-safe: a {@code Net<Tensor>} can produce a {@code Net<Var>} without an
   * unchecked cast. The returned gradients have the original parameter tree's
   * runtime shape and {@code Tensor} leaves.
   *
   * <p>A record containing parameter leaves must use a generic leaf component
   * (for example, {@code record Net<T>(T weight, T bias)}) so the factory can
   * create the corresponding {@code Net<Var>}.
   */
  public static <P, V> ValueAndGradTree<P> valueAndGrad(
      Function<? super V, Var> lossFn,
      P paramTree,
      Function<? super List<Var>, ? extends V> variableTreeFactory) {
    Objects.requireNonNull(lossFn, "lossFn");
    Objects.requireNonNull(variableTreeFactory, "variableTreeFactory");

    TreeUtil.Flattened flattened = TreeUtil.flatten(paramTree);
    List<Var> paramVars = new ArrayList<>(flattened.leaves().size());
    for (Object leaf : flattened.leaves()) {
      if (!(leaf instanceof Tensor tensor)) {
        throw new IllegalArgumentException(
            "structured AutoDiff parameters must have Tensor leaves, found "
                + (leaf == null ? "null" : leaf.getClass().getName()));
      }
      paramVars.add(Var.leaf(tensor));
    }

    V variableTree = variableTreeFactory.apply(List.copyOf(paramVars));
    TreeUtil.Flattened flattenedVariables = TreeUtil.flatten(variableTree);
    if (!variableTreeDef(flattened.def()).equals(flattenedVariables.def())) {
      throw new IllegalArgumentException("variable tree factory must preserve the parameter tree structure");
    }
    for (int i = 0; i < paramVars.size(); i++) {
      if (flattenedVariables.leaves().get(i) != paramVars.get(i)) {
        throw new IllegalArgumentException(
            "variable tree factory must preserve each parameter variable's position");
      }
    }

    ValueAndGrad result = valueAndGradFlat(vars -> lossFn.apply(variableTree), paramVars);
    Object gradientTree = TreeUtil.unflatten(flattened.def(), result.grads());
    return new ValueAndGradTree<>(result.value(), checkedTreeType(paramTree, gradientTree));
  }

  public static ValueAndGrad valueAndGrad(Function<List<Var>, Var> lossFn, List<Tensor> params) {
    Objects.requireNonNull(lossFn, "lossFn");
    Objects.requireNonNull(params, "params");
    List<Var> paramVars = new ArrayList<>(params.size());
    for (Tensor p : params) {
      paramVars.add(Var.leaf(p));
    }
    return valueAndGradFlat(lossFn, paramVars);
  }

  private static ValueAndGrad valueAndGradFlat(Function<? super List<Var>, Var> lossFn, List<Var> paramVars) {
    List<Var> trace = Tape.begin();
    java.util.Set<Tensor> retained = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
    for (Var parameter : paramVars) retained.add(parameter.value());
    try {
      Var loss = lossFn.apply(paramVars);
      loss.backward();
      List<Tensor> grads = new ArrayList<>(paramVars.size());
      for (Var parameter : paramVars) {
        Tensor grad = parameter.grad();
        grads.add(grad != null ? grad : parameter.value().fullLike(0f));
      }
      retained.add(loss.value());
      retained.addAll(grads);
      return new ValueAndGrad(loss.value(), List.copyOf(grads));
    } finally {
      java.util.Set<Tensor> released = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
      for (Var variable : trace) {
        if (!retained.contains(variable.value()) && released.add(variable.value())) variable.value().close();
        if (variable.grad() != null && !retained.contains(variable.grad()) && released.add(variable.grad())) variable.grad().close();
      }
      for (Var parameter : paramVars) {
        if (parameter.grad() != null && !retained.contains(parameter.grad()) && released.add(parameter.grad())) parameter.grad().close();
      }
      Tape.end();
    }
  }

  private static <T> T checkedTreeType(T template, Object tree) {
    if (template == null) {
      if (tree != null) {
        throw new IllegalArgumentException("rebuilt null parameter tree was not null");
      }
      return null;
    }
    if (!template.getClass().isInstance(tree)) {
      throw new IllegalArgumentException("rebuilt gradient tree has unexpected type "
          + (tree == null ? "null" : tree.getClass().getName()));
    }
    @SuppressWarnings("unchecked")
    T typedTree = (T) tree;
    return typedTree;
  }

  @SuppressWarnings("unchecked")
  private static <V> V variableTree(TreeDef def, List<Var> variables) {
    return (V) TreeUtil.unflatten(def, variables);
  }

  private static TreeDef variableTreeDef(TreeDef def) {
    return def.mapArrayTypes(AutoDiff::variableArrayComponentType);
  }

  private static Class<?> variableArrayComponentType(Class<?> type) {
    if (type == Tensor.class) {
      return Var.class;
    }
    if (type.isArray()) {
      return Array.newInstance(variableArrayComponentType(type.getComponentType()), 0).getClass();
    }
    return type;
  }

  /** Gradient-only convenience over {@link #valueAndGrad}; discards the loss value. */
  public static List<Tensor> grad(Function<List<Var>, Var> lossFn, List<Tensor> params) {
    ValueAndGrad result = valueAndGrad(lossFn, params);
    result.value().close();
    return result.grads();
  }
}
