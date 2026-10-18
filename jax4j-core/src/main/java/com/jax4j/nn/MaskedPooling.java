package com.jax4j.nn;
import com.jax4j.tensor.Tensor;
import com.jax4j.autodiff.Var;
/** Presence-aware mean/max pooling over enemy axis 1; defined output for zero enemies.
 * Constants must remain open through the autodiff backward sweep.
 */
public final class MaskedPooling implements AutoCloseable {
    private final Tensor present,count,denominator,penalty,active;
    public MaskedPooling(Tensor mask){
        if(mask.shape().rank()!=3||mask.shape().dim(2)!=1)throw new IllegalArgumentException("Expected [batch,enemies,1] presence mask");
        for(float value:mask.toFloatArray())if(value!=0&&value!=1)throw new IllegalArgumentException("Presence is explicitly zero or one");
        present=mask.mul(1);count=present.sum(1);denominator=count.maximum(1);active=count.minimum(1);
        try(Tensor absent=present.sub(1)){penalty=absent.mul(1e30);}
    }
    public record Result(Tensor mean,Tensor maximum,Tensor count) implements AutoCloseable {
        public void close(){mean.close();maximum.close();count.close();}
    }
    public record DifferentiableResult(Var mean,Var maximum){}
    public Result forward(Tensor encoded){
        check(encoded);
        try(Tensor masked=encoded.mul(present);Tensor sum=masked.sum(1);Tensor penalized=encoded.add(penalty);Tensor max=penalized.max(1);Tensor gated=max.mul(active)){
            return new Result(sum.div(denominator),gated.add(0),count.mul(1));
        }
    }
    public DifferentiableResult forward(Var encoded){
        check(encoded.value());
        return new DifferentiableResult(encoded.mul(Var.leaf(present)).sum(1).div(Var.leaf(denominator)),encoded.add(Var.leaf(penalty)).max(1).mul(Var.leaf(active)).add(Var.leaf(active).scaled(0)));
    }
    private void check(Tensor encoded){if(encoded.shape().rank()!=3||encoded.shape().dim(0)!=present.shape().dim(0)||encoded.shape().dim(1)!=present.shape().dim(1))throw new IllegalArgumentException("Encoded heads and mask shape mismatch");}
    public void close(){present.close();count.close();denominator.close();penalty.close();active.close();}
}
