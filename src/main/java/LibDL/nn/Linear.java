package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Tensor;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class Linear extends LayerTensor {

    private final Constant W;
    private final Constant B;
    private final boolean bias;

    @Override
    protected Tensor core() {
        if (bias) return input.mm(W).add(B);
        else return input.mm(W);
    }

    @JsonCreator
    public Linear(@JsonProperty("in_features")int in_features, @JsonProperty("out_features")int out_features) {
        this(in_features, out_features, true);
    }

    @JsonCreator
    public Linear(@JsonProperty("in_features")int in_features, @JsonProperty("out_features")int out_features, @JsonProperty("bias")boolean bias) {
        W = new Constant(Nd4j.create(in_features, out_features), true);
        if (bias) {
            B = new Constant(Nd4j.create(1, out_features), true);
        } else {
            B = null;
        }
        this.bias = bias;

        resetParameters();
    }

    private void resetParameters() {
        long fanIn = W.value.size(0);
        WeightInit.kaimingUniform(W.value, Math.sqrt(5));
        if(bias) {
            double bound = 1 / Math.sqrt(fanIn);
            WeightInit.uniform(B.value, -bound, bound);
        }
    }

    @Override
    public String toString(){
        return "Constant W:\n" + W.value + "\n"
                               + Arrays.toString(W.parameters()) + "\n"
                               + W.toString() + "\n" +
               "Constant B:\n" + B.value + "\n"
                               + Arrays.toString(B.parameters()) + "\n"
                               + B.toString() + "\n" +
               "bias:"         + bias + "\n" +
               "out:"          + this.out + "\n" +
               "dout:"         + this.dout + "\n" +
               "input:"        + this.input + "\n";
    }
}
