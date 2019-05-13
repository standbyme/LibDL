package LibDL.nn;

import LibDL.Tensor.Parameter;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static LibDL.nn.RNNBase.RNNType.TYPE_GRU;

public class GRU extends RNNBase {

    public GRU(int inputSize, int hiddenSize, int numLayers) {
        super(inputSize, hiddenSize, numLayers,
                true, false, false,
                0, false, TYPE_GRU);
    }

    @Override
    protected Tensor[] rnn_impl(Tensor input, Tensor[] outList, Tensor prevHidden, int seqLen, Tensor prev_cell, int currLayer) {
        for (int i = 0; i < seqLen; i++) {
            Tensor currIn = input.get(i);
            Tensor r = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden,
                            weight_ih[pm(currLayer, PARAM_R)],
                            weight_hh[pm(currLayer, PARAM_R)],
                            bias_hh[pm(currLayer, PARAM_R)],
                            bias_ih[pm(currLayer, PARAM_R)], null)
            );
            Tensor currOut = calculate_gate(currIn, prevHidden,
                    weight_ih[pm(currLayer, PARAM_N)],
                    weight_hh[pm(currLayer, PARAM_N)],
                    bias_hh[pm(currLayer, PARAM_N)],
                    bias_ih[pm(currLayer, PARAM_N)], r);

            Tensor u = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden,
                            weight_ih[pm(currLayer, PARAM_Z)],
                            weight_hh[pm(currLayer, PARAM_Z)],
                            bias_hh[pm(currLayer, PARAM_Z)],
                            bias_ih[pm(currLayer, PARAM_Z)], null)
            );

            currOut = compute_current(currOut);

            outList[i] = u.mul(prevHidden).add(Tensor.ones(u.sizes()).sub(u).mul(currOut));
            prevHidden = outList[i];
            h_n [currLayer]= prevHidden;
        }
        return outList;
    }


}
