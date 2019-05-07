package LibDL.nn;

import LibDL.Tensor.Tensor;

public class LSTM extends RNNAuto {
    public LSTM(int inputSize, int hiddenSize) {
        super(inputSize, hiddenSize, false, false, TYPE_LSTM);

    }

    @Override
    protected Tensor[] rnn_impl(Tensor input, Tensor[] outList, Tensor prevHidden, int seqLen, Tensor prev_cell) {
        for (int i = 0; i < seqLen; i++) {
            Tensor currIn = input.get(i);
            Tensor currOut = calculate_gate(currIn, prevHidden, weight_ih, weight_hh, bias_hh, bias_ih, null);

            currOut = compute_current(currOut);

            Tensor i_t = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, gu_weight_ih, gu_weight_hh, gu_bias_hh, gu_bias_ih, null)
            );
            Tensor f_t = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, gf_weight_ih, gf_weight_hh, gf_bias_hh, gf_bias_ih, null)
            );
            Tensor o_t = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, gro_weight_ih, gro_weight_hh, gro_bias_hh, gro_bias_ih, null)
            );

            prev_cell = f_t.mul(prev_cell).add(i_t.mul(currOut));

            prevHidden = outList[i] = o_t.mul(Tensor.tanh(prev_cell));
        }
        return outList;
    }


}
