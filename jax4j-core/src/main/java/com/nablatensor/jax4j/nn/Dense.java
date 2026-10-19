package com.nablatensor.jax4j.nn;
import com.nablatensor.jax4j.tensor.Tensor;
import com.nablatensor.jax4j.autodiff.Var;
/** Shared dense layer over the final dimension of any rank, including [B,8,19]. */
public final class Dense {
    private Dense(){}
    public static Tensor forward(Tensor input,Tensor weight,Tensor bias){
        int[] dims=input.shape().dims().clone();
        if(dims.length==0||weight.shape().rank()!=2||dims[dims.length-1]!=weight.shape().dim(0))throw new IllegalArgumentException("Dense shape mismatch");
        int rows=Math.toIntExact(input.shape().size()/dims[dims.length-1]);dims[dims.length-1]=weight.shape().dim(1);
        try(Tensor flat=input.reshape(rows,weight.shape().dim(0));Tensor product=flat.matmul(weight);Tensor shifted=product.add(bias)){
            return shifted.reshape(dims);
        }
    }
    public static Var forward(Var input,Var weight,Var bias){
        int[] dims=input.value().shape().dims().clone();
        if(dims.length==0||weight.value().shape().rank()!=2||dims[dims.length-1]!=weight.value().shape().dim(0))throw new IllegalArgumentException("Dense shape mismatch");
        int rows=Math.toIntExact(input.value().shape().size()/dims[dims.length-1]);dims[dims.length-1]=weight.value().shape().dim(1);
        return input.reshape(rows,weight.value().shape().dim(0)).matmul(weight).add(bias).reshape(dims);
    }
    /** Allocation-free float32 inference kernel, row-major shared weights. Output must not alias input. */
    public static void into(float[] input,int rows,int in,int out,float[] weight,float[] bias,float[] output,boolean relu){
        if(input.length<rows*in||output.length<rows*out||weight.length!=in*out||bias.length!=out)throw new IllegalArgumentException("Dense buffer dimensions");
        for(int r=0;r<rows;r++){
            int oi=r*out;System.arraycopy(bias,0,output,oi,out);
            for(int k=0;k<in;k++){float x=input[r*in+k];int wi=k*out;for(int j=0;j<out;j++)output[oi+j]+=x*weight[wi+j];}
            if(relu)for(int j=0;j<out;j++)output[oi+j]=Math.max(0,output[oi+j]);
        }
    }
}
