package LibDL.nn;

import LibDL.Tensor.Tensor;

import static LibDL.nn.RNNBase.RNNType.TYPE_GRU;

public class GRU extends RNNBase {

    public GRU(int inputSize, int hiddenSize, int numLayers) {
        this(inputSize, hiddenSize, numLayers, false, true, 0);
    }

    public GRU(int inputSize, int hiddenSize, int numLayers, boolean relu, boolean bias, double dropout) {
        super(inputSize, hiddenSize, numLayers,
                bias, relu, false,
                dropout, false, TYPE_GRU);
    }

    @Override
    protected Tensor[] rnn_impl(Tensor input, Tensor[] outList, Tensor prevHidden, int seqLen, Tensor prev_cell, int currLayer) {
        for (int i = 0; i < seqLen; i++) {
            Tensor currIn = input.get(i);
            Tensor r = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, currLayer, PARAM_R, null)
            );
            Tensor currOut = calculate_gate(currIn, prevHidden, currLayer, PARAM_N, r);

            Tensor u = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, currLayer, PARAM_Z, null)
            );

            currOut = compute_current(currOut);

            outList[i] = u.mul(prevHidden).add(Tensor.ones(u.sizes()).sub(u).mul(currOut));
            prevHidden = outList[i];
            h_n[currLayer] = prevHidden;
        }
        return outList;
    }


}
