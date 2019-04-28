package LibDL.nn;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Parameter;
import org.nd4j.linalg.factory.Nd4j;

public class Dense extends Module {

    private Parameter W;
    private Parameter B;
    private int in_features;
    private int out_features;
    private boolean bias;


    public Dense(int in_features, int out_features) {
        this(in_features, out_features, true);
    }

    public Dense(int in_features, int out_features, boolean bias) {
        this.in_features = in_features;
        this.out_features = out_features;
        this.bias = bias;

        W = new Parameter(Nd4j.create(in_features, out_features));
        if (bias) {
            B = new Parameter(Nd4j.create(1, out_features));
        } else {
            B = null;
        }

        resetParameters();
    }

    @Override
    public Tensor forward(Tensor input) {
        if (bias) return (input.mm(W).addVector(B));
        else return (input.mm(W));
    }

    private void resetParameters() {
        long fanIn = W.data.size(0);
        WeightInit.kaimingUniform(W.data, Math.sqrt(5));
        if(bias) {
            double bound = 1 / Math.sqrt(fanIn);
            WeightInit.uniform(B.data, -bound, bound);
        }
    }

    @Override
    public String toString() {
        return "Dense(in_features=" + in_features + ", out_features=" + out_features
                + ", bias=" + bias + ")";
    }
}
