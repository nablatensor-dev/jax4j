package com.nablatensor.jax4j.nn;
import com.nablatensor.jax4j.tensor.*;
import com.nablatensor.jax4j.autodiff.*;
import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;
class MaskedPoolingTest {
    @Test void masksJunkPreservesNegativeMaximaAndBackpropagatesOnlyPresentHeads(){
        try(Tensor x=Tensors.array(new float[]{-2,3,999,999,-1,4},1,3,2);Tensor mask=Tensors.array(new float[]{1,0,1},1,3,1);MaskedPooling pool=new MaskedPooling(mask);var result=pool.forward(x)){
            assertArrayEquals(new float[]{-1.5f,3.5f},result.mean().toFloatArray());assertArrayEquals(new float[]{-1,4},result.maximum().toFloatArray());assertEquals(2,result.count().item());
            var vg=AutoDiff.valueAndGrad(p->{var parts=pool.forward(p.get(0));return parts.mean().sum().add(parts.maximum().sum());},List.of(x));
            assertArrayEquals(new float[]{.5f,.5f,0,0,1.5f,1.5f},vg.grads().get(0).toFloatArray());vg.value().close();vg.grads().forEach(Tensor::close);
        }
    }
    @Test void emptySceneHasFiniteZeroOutput(){
        try(Tensor x=Tensors.array(new float[]{2,3,4,5},1,2,2);Tensor mask=Tensors.zeros(1,2,1);MaskedPooling pool=new MaskedPooling(mask);var result=pool.forward(x)){
            assertArrayEquals(new float[]{0,0},result.mean().toFloatArray());assertArrayEquals(new float[]{0,0},result.maximum().toFloatArray());assertEquals(0,result.count().item());
        }
    }
}
