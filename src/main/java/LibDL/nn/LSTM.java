package LibDL.nn;

import LibDL.Tensor.Parameter;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static LibDL.nn.RNNBase.RNNType.TYPE_LSTM;

public class LSTM extends RNNBase {

    public Tensor[] c_n;

    public LSTM(int inputSize, int hiddenSize, int numLayers) {
        super(inputSize, hiddenSize, numLayers,
                true, false, false,
                0, false, TYPE_LSTM);
        c_n=new Tensor[numLayers];
    }

    @Override
    protected Tensor[] rnn_impl(Tensor input, Tensor[] outList, Tensor prevHidden, int seqLen, Tensor prev_cell, int currLayer) {
        for (int i = 0; i < seqLen; i++) {
            Tensor currIn = input.get(i);
            Tensor g_t = calculate_gate(currIn, prevHidden, weight_ih[pm(currLayer, PARAM_G)], weight_hh[pm(currLayer, PARAM_G)], bias_hh[pm(currLayer, PARAM_G)], bias_ih[pm(currLayer, PARAM_G)], null);

            g_t = compute_current(g_t);

            Tensor i_t = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, weight_ih[pm(currLayer, PARAM_I)], weight_hh[pm(currLayer, PARAM_I)], bias_hh[pm(currLayer, PARAM_I)], bias_ih[pm(currLayer, PARAM_I)], null)
            );
            Tensor f_t = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, weight_ih[pm(currLayer, PARAM_F)], weight_hh[pm(currLayer, PARAM_F)], bias_hh[pm(currLayer, PARAM_F)], bias_ih[pm(currLayer, PARAM_F)], null)
            );
            Tensor o_t = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, weight_ih[pm(currLayer, PARAM_O)], weight_hh[pm(currLayer, PARAM_O)], bias_hh[pm(currLayer, PARAM_O)], bias_ih[pm(currLayer, PARAM_O)], null)
            );

            Tensor new_cell = f_t.mul(prev_cell).add(i_t.mul(g_t));


            h_n [currLayer]= prevHidden = outList[i] = o_t.mul(Tensor.tanh(new_cell));
            c_n [currLayer] = prev_cell = new_cell;
        }
        return outList;
    }

    public void setParam(int param_type, INDArray... params) {
        Parameter[] paramList = null;
        for (int currLayer = 0; currLayer < numLayers; currLayer++) {
            INDArray param = params[currLayer];
            switch (param_type) {
                case WEIGHT_HH:
                    paramList = new Parameter[]{weight_hh[pm(currLayer, PARAM_I)], weight_hh[pm(currLayer, PARAM_F)], weight_hh[pm(currLayer, PARAM_G)], weight_hh[pm(currLayer, PARAM_O)]};
                    break;
                case WEIGHT_IH:
                    paramList = new Parameter[]{weight_ih[pm(currLayer, PARAM_I)], weight_ih[pm(currLayer, PARAM_F)], weight_ih[pm(currLayer, PARAM_G)], weight_ih[pm(currLayer, PARAM_O)]};
                    break;
                case BIAS_HH:
                    paramList = new Parameter[]{bias_hh[pm(currLayer, PARAM_I)], bias_hh[pm(currLayer, PARAM_F)], bias_hh[pm(currLayer, PARAM_G)], bias_hh[pm(currLayer, PARAM_O)]};
                    break;
                case BIAS_IH:
                    paramList = new Parameter[]{bias_ih[pm(currLayer, PARAM_I)], bias_ih[pm(currLayer, PARAM_F)], bias_ih[pm(currLayer, PARAM_G)], bias_ih[pm(currLayer, PARAM_O)]};
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
