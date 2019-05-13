package LibDL.nn;

import LibDL.Tensor.Tensor;

import static LibDL.nn.RNNBase.RNNType.TYPE_RNN;

public class RNN extends RNNBase {
    // Layer parameters

    public RNN(int inputSize, int hiddenSize, int numLayers) {
        this(inputSize, hiddenSize, numLayers, false, true, 0);
    }

    public RNN(int inputSize, int hiddenSize, int numLayers, boolean relu, boolean bias, double dropout) {
        super(inputSize, hiddenSize, numLayers,
                bias, relu, false,
                dropout, false, TYPE_RNN);
    }

    protected Tensor[] rnn_impl(Tensor input, Tensor[] outList, Tensor prevHidden, int seqLen, Tensor prev_cell, int currLayer) {
        for (int i = 0; i < seqLen; i++) {
            Tensor currIn = input.get(i);
            Tensor currOut = calculate_gate(currIn, prevHidden, currLayer, PARAM_I, null);

            currOut = compute_current(currOut);

            outList[i] = prevHidden = h_n[currLayer] = currOut;
        }
        return outList;
    }


}
