package com.nablatensor.jax4j.nn;
import com.nablatensor.jax4j.tensor.*;
import com.nablatensor.jax4j.autodiff.*;
import java.util.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
class DenseTest {
    @Test void sharedThreeDimensionalLayerAndGradients(){
        try(Tensor x=Tensors.array(new float[]{1,2,3,4},1,2,2);Tensor w=Tensors.array(new float[]{2,3},2,1);Tensor b=Tensors.array(new float[]{1},1);Tensor output=Dense.forward(x,w,b)){
            assertArrayEquals(new int[]{1,2,1},output.shape().dims());assertArrayEquals(new float[]{9,19},output.toFloatArray());
            var vg=AutoDiff.valueAndGrad(p->Dense.forward(Var.leaf(x),p.get(0),p.get(1)).sum(),List.of(w,b));
            assertArrayEquals(new float[]{4,6},vg.grads().get(0).toFloatArray());assertArrayEquals(new float[]{2},vg.grads().get(1).toFloatArray());
            vg.value().close();vg.grads().forEach(Tensor::close);
        }
    }
    @Test void optimizerLearnsSmallRegression(){
        try(Tensor x=Tensors.array(new float[]{1,2,3,4},4,1);Tensor y=Tensors.array(new float[]{3,5,7,9},4,1)){
            List<Tensor> p=List.of(Tensors.zeros(1,1),Tensors.zeros(1));Adam adam=new Adam(.05f);float first=0,last=0;
            for(int i=0;i<120;i++){
                var vg=AutoDiff.valueAndGrad(params->{Var error=Dense.forward(Var.leaf(x),params.get(0),params.get(1)).sub(Var.leaf(y));return error.mul(error).mean();},p);
                last=vg.value().item();if(i==0)first=last;List<Tensor> next=adam.update(p,vg.grads());p.forEach(Tensor::close);vg.grads().forEach(Tensor::close);vg.value().close();p=next;
            }
            assertTrue(last<first*.01,"Regression loss should decrease substantially");p.forEach(Tensor::close);
        }
    }
}
