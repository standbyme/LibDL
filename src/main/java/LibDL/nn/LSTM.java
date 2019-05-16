package LibDL.nn;

import LibDL.Tensor.Tensor;

import static LibDL.nn.RNNBase.RNNType.TYPE_LSTM;

public class LSTM extends RNNBase {

    public Tensor[] c_n;


    public LSTM(int inputSize, int hiddenSize, int numLayers) {
        this(inputSize, hiddenSize, numLayers,false, true, 0);
    }

    public LSTM(int inputSize, int hiddenSize, int numLayers, boolean relu, boolean bias, double dropout) {
        super(inputSize, hiddenSize, numLayers,
                bias, relu, false,
                0, false, TYPE_LSTM);
        c_n = new Tensor[numLayers];
    }

    @Override
    protected Tensor[] rnn_impl(Tensor input, Tensor[] outList, Tensor prevHidden, int seqLen, Tensor prev_cell, int currLayer) {
        for (int i = 0; i < seqLen; i++) {
            Tensor currIn = input.get(i);
            Tensor g_t = calculate_gate(currIn, prevHidden, currLayer, PARAM_G, null);

            g_t = compute_current(g_t);

            Tensor i_t = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, currLayer, PARAM_I, null)
            );
            Tensor f_t = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, currLayer, PARAM_F, null)
            );
            Tensor o_t = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, currLayer, PARAM_O, null)
            );

            Tensor new_cell = f_t.mul(prev_cell).add(i_t.mul(g_t));


            h_n[currLayer] = prevHidden = outList[i] = o_t.mul(Tensor.tanh(new_cell));
            c_n[currLayer] = prev_cell = new_cell;
        }
        return outList;
    }


}
