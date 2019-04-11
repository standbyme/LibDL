package LibDL.nn;

import LibDL.Tensor.Variable;
import LibDL.Tensor.LayerTensor;
import org.nd4j.linalg.factory.Nd4j;

public class Dense extends LayerTensor {

    private final Variable W;
    private final Variable B;
    private final boolean bias;


    public Dense(int in_features, int out_features) {
        this(in_features, out_features, true);
    }

    public Dense(int in_features, int out_features, boolean bias) {
//        super(false);
        W = new Variable(Nd4j.create(in_features, out_features), true);
        if (bias) {
            B = new Variable(Nd4j.create(1, out_features), true);
        } else {
            B = null;
        }
        this.bias = bias;

        resetParameters();

        if (bias) setCore(input.mm(W).add(B));
        else setCore(input.mm(W));
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
