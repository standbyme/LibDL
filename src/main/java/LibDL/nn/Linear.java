package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.factory.Nd4j;

public class Linear extends LayerTensor {

    private final Constant W;
    private final Constant B;
    private final boolean bias;

    @Override
    protected Tensor core() {
        if (bias) return X.mm(W).add(B);
        else return X.mm(W);
    }

    public Linear(int in_features, int out_features) {
        this(in_features, out_features, true);
    }

    public Linear(int in_features, int out_features, boolean bias) {
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
