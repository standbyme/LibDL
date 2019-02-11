package LibDL.Tensor.Layer;

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
        W = new Constant(Nd4j.ones(in_features, out_features), true);
        B = new Constant(Nd4j.ones(1, out_features), true);
        bias = true;
    }

    public Linear(int in_features, int out_features, boolean bias) {
        W = new Constant(Nd4j.ones(in_features, out_features), true);
        if (bias) {
            B = new Constant(Nd4j.ones(1, out_features), true);
        } else {
            B = null;
        }
        this.bias = bias;
    }
}
