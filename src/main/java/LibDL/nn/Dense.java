package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.factory.Nd4j;

public class Dense extends LayerTensor {

    private final Constant W;
    private final Constant B;
    private final boolean bias;

    @Override
    protected Tensor core() {
        if (bias) return input.mm(W).add(B);
        else return input.mm(W);
    }

    public Dense(int in_features, int out_features) {
        this(in_features, out_features, true);
    }

    public Dense(int in_features, int out_features, boolean bias) {
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
}
