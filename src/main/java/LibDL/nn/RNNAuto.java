package LibDL.nn;

import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import static LibDL.nn.RNNBase.RNNType.TYPE_RNN;

public class RNNAuto extends RNNBase {
    // Layer parameters


    public RNNAuto(int inputSize, int hiddenSize, int numLayers) {
        super(inputSize, hiddenSize, numLayers,
                true, false, false,
                0, false, TYPE_RNN);
    }

    protected Tensor[] rnn_impl(Tensor input, Tensor[] outList, Tensor prevHidden, int seqLen, Tensor prev_cell, int currLayer) {
        for (int i = 0; i < seqLen; i++) {
            Tensor currIn = input.get(i);
            Tensor currOut = calculate_gate(currIn, prevHidden,
                    weight_ih[pm(currLayer, PARAM_I)],
                    weight_hh[pm(currLayer, PARAM_I)],
                    bias_hh[pm(currLayer, PARAM_I)],
                    bias_ih[pm(currLayer, PARAM_I)], null);

            currOut = compute_current(currOut);

            outList[i] = prevHidden = h_n = currOut;
        }
        return outList;
    }

    public void setParam(int param_type, INDArray... param) {
        for (int currLayer = 0; currLayer < numLayers; currLayer++) {
            switch (param_type) {
                case WEIGHT_HH:
                    weight_hh[pm(currLayer, PARAM_I)].data = param[currLayer];
                    break;
                case WEIGHT_IH:
                    weight_ih[pm(currLayer, PARAM_I)].data = param[currLayer];
                    break;
                case BIAS_HH:
                    bias_hh[pm(currLayer, PARAM_I)].data = param[currLayer];
                    break;
                case BIAS_IH:
                    bias_ih[pm(currLayer, PARAM_I)].data = param[currLayer];
            }
        }
    }

}
