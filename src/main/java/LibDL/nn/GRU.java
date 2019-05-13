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
                    calculate_gate(currIn, prevHidden, weight_ih[pm(currLayer, PARAM_R)], weight_hh[pm(currLayer, PARAM_R)], bias_hh[pm(currLayer, PARAM_R)], bias_ih[pm(currLayer, PARAM_R)], null)
            );
            Tensor currOut = calculate_gate(currIn, prevHidden, weight_ih[pm(currLayer, PARAM_N)], weight_hh[pm(currLayer, PARAM_N)], bias_hh[pm(currLayer, PARAM_N)], bias_ih[pm(currLayer, PARAM_N)], r);

            Tensor u = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, weight_ih[pm(currLayer, PARAM_Z)], weight_hh[pm(currLayer, PARAM_Z)], bias_hh[pm(currLayer, PARAM_Z)], bias_ih[pm(currLayer, PARAM_Z)], null)
            );

            currOut = compute_current(currOut);

            outList[i] = u.mul(prevHidden).add(Tensor.ones(u.sizes()).sub(u).mul(currOut));
            prevHidden = outList[i];
            h_n [currLayer]= prevHidden;
        }
        return outList;
    }

    public void setParam(int param_type, INDArray... params) {
        Parameter[] paramList = null;
        for (int currLayer = 0; currLayer < numLayers; currLayer++) {
            INDArray param = params[currLayer];
            switch (param_type) {
                case WEIGHT_HH:
                    paramList = new Parameter[]{weight_hh[pm(currLayer, PARAM_R)], weight_hh[pm(currLayer, PARAM_Z)], weight_hh[pm(currLayer, PARAM_N)]};
                    break;
                case WEIGHT_IH:
                    paramList = new Parameter[]{weight_ih[pm(currLayer, PARAM_R)], weight_ih[pm(currLayer, PARAM_Z)], weight_ih[pm(currLayer, PARAM_N)]};
                    break;
                case BIAS_HH:
                    paramList = new Parameter[]{bias_hh[pm(currLayer, PARAM_R)], bias_hh[pm(currLayer, PARAM_Z)], bias_hh[pm(currLayer, PARAM_N)]};
                    break;
                case BIAS_IH:
                    paramList = new Parameter[]{bias_ih[pm(currLayer, PARAM_R)], bias_ih[pm(currLayer, PARAM_Z)], bias_ih[pm(currLayer, PARAM_N)]};
            }

            INDArrayIndex[] indices = new INDArrayIndex[param.rank()];
            for (int i = 1; i < indices.length; i++) {
                indices[i] = NDArrayIndex.all();
            }

            for (int i = 0; i < rnn_type.gateSize(); i++) {
                indices[0] = NDArrayIndex.interval(i * hiddenSize, i * hiddenSize + hiddenSize);
                assert paramList != null;
                paramList[i].data = param.get(indices);
            }
        }
    }

}
