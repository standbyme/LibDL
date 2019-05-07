package LibDL.nn;

import LibDL.Tensor.Tensor;

public class GRU extends RNNAuto {
    public GRU(int inputSize, int hiddenSize) {
        super(inputSize, hiddenSize);
        init_params(gu_weight_hh, gu_bias_ih, gu_bias_hh, gu_bias_ih);
        init_params(gro_weight_hh, gro_bias_ih, gro_bias_hh, gro_bias_ih);
    }

    @Override
    protected void forward_loop(Tensor input, Tensor[] outList, Tensor prevHidden, int seqLen) {

        for (int i = 0; i < seqLen; i++) {
            Tensor currIn = input.get(i);
            Tensor r = calculate_gate(currIn, prevHidden, gro_weight_ih, gro_weight_hh, bias_hh, bias_ih, null);


            Tensor currOut = calculate_gate(currIn, prevHidden, weight_ih, weight_hh, bias_hh, bias_ih, r);

            currOut = compute_current(currOut);

            Tensor u = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, gu_weight_ih, gu_weight_hh, gu_bias_hh, gu_bias_ih, null)
            );
            outList[i] = u.mul(currOut)
                    .add(Tensor.ones(u.sizes())
                            .sub(u).mul(prevHidden));
            prevHidden = currOut;
        }
    }

}
