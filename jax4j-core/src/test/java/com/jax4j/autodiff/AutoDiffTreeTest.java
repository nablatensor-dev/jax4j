package com.jax4j.autodiff;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.jax4j.tensor.Tensors;
import com.jax4j.tensor.Tensor;
import com.jax4j.tensor.tree.TreeUtil;

import java.util.List;

import org.junit.jupiter.api.Test;

class AutoDiffTreeTest {

  record Pair<T>(T left, T right) {
  }

  @Test
  void preservesGenericRecordShapeForStructuredParameters() {
    Pair<Tensor> params = new Pair<>(Tensors.array(new float[] {2}), Tensors.array(new float[] {3}));
    AutoDiff.ValueAndGradTree<Pair<Tensor>> result = AutoDiff.valueAndGrad(
        (Pair<Var> variables) -> variables.left().mul(variables.right()).sum(),
        params);
    try {
      assertEquals(TreeUtil.structure(params), TreeUtil.structure(result.grads()));
      assertArrayEquals(new float[] {3}, result.grads().left().toFloatArray());
      assertArrayEquals(new float[] {2}, result.grads().right().toFloatArray());
    } finally {
      result.value().close();
      TreeUtil.closeAll(result.grads());
      TreeUtil.closeAll(params);
    }
  }

  @Test
  void retainsFlatListValueAndGradOverload() {
    Tensor parameter = Tensors.array(new float[] {4});
    AutoDiff.ValueAndGrad result = AutoDiff.valueAndGrad(vars -> vars.getFirst().mul(vars.getFirst()).sum(),
        List.of(parameter));
    try {
      assertArrayEquals(new float[] {8}, result.grads().getFirst().toFloatArray());
    } finally {
      result.value().close();
      TreeUtil.closeAll(result.grads());
      parameter.close();
    }
  }

  @Test
  void unusedStructuredParametersReceiveZeroGradients() {
    Pair<Tensor> params = new Pair<>(
        Tensors.array(new float[] {2}), Tensors.array(new float[] {3, 4}));
    AutoDiff.ValueAndGradTree<Pair<Tensor>> result = AutoDiff.valueAndGrad(
        (Pair<Var> variables) -> variables.left().sum(),
        params);
    try {
      assertArrayEquals(new float[] {1}, result.grads().left().toFloatArray());
      assertArrayEquals(new float[] {0, 0}, result.grads().right().toFloatArray());
    } finally {
      result.value().close();
      TreeUtil.closeAll(result.grads());
      TreeUtil.closeAll(params);
    }
  }

  @Test
  void preservesTensorArrayShapeWithVarArrayDuringDifferentiation() {
    Tensor[] params = {Tensors.array(new float[] {2}), Tensors.array(new float[] {3})};
    AutoDiff.ValueAndGradTree<Tensor[]> result = AutoDiff.valueAndGrad(
        (Var[] variables) -> variables[0].mul(variables[1]).sum(),
        params);
    try {
      assertArrayEquals(new float[] {3}, result.grads()[0].toFloatArray());
      assertArrayEquals(new float[] {2}, result.grads()[1].toFloatArray());
    } finally {
      result.value().close();
      TreeUtil.closeAll(result.grads());
      TreeUtil.closeAll(params);
    }
  }

  @Test
  void rejectsNonTensorLeavesAndVariableFactoriesThatChangeStructure() {
    assertThrows(IllegalArgumentException.class,
        () -> AutoDiff.valueAndGrad((Pair<Var> variables) -> variables.left().sum(),
            new Pair<>("not-a-tensor", "also-not-a-tensor"),
            vars -> new Pair<>(vars.get(0), vars.get(1))));

    Pair<Tensor> params = new Pair<>(Tensors.array(new float[] {2}), Tensors.array(new float[] {3}));
    try {
      assertThrows(IllegalArgumentException.class,
          () -> AutoDiff.valueAndGrad((List<Var> variables) -> variables.getFirst().sum(),
              params,
              List::copyOf));
      assertThrows(IllegalArgumentException.class,
          () -> AutoDiff.valueAndGrad((Pair<Var> variables) -> variables.left().sum(),
              params,
              vars -> new Pair<>(vars.get(1), vars.get(0))));
    } finally {
      TreeUtil.closeAll(params);
    }
  }
}
