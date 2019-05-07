package LibDL.nn;

import LibDL.Tensor.Tensor;

public class LSTM extends RNNAuto {
    public LSTM(int inputSize, int hiddenSize) {
        super(inputSize, hiddenSize, false, false, TYPE_LSTM);

    }

    @Override
    protected void rnn_impl(Tensor input, Tensor[] outList, Tensor prevHidden, int seqLen) {
        for (int i = 0; i < seqLen; i++) {
            Tensor currIn = input.get(i);
            Tensor currOut = calculate_gate(currIn, prevHidden, weight_ih, weight_hh, bias_hh, bias_ih, null);

            currOut = compute_current(currOut);

            Tensor u = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, gu_weight_ih, gu_weight_hh, gu_bias_hh, gu_bias_ih, null)
            );
            Tensor f = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, gf_weight_ih, gf_weight_hh, gf_bias_hh, gf_bias_ih, null)
            );
            Tensor o = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, gro_weight_ih, gro_weight_hh, gro_bias_hh, gro_bias_ih, null)
            );

            outList[i] = o.mul(u.mul(currOut).add(f.mul(currOut)));

            prevHidden = currOut;
        }
    }

}
