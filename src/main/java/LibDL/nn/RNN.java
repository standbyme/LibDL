package LibDL.nn;

import LibDL.Tensor.Parameter;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static LibDL.nn.RNNBase.RNNType.TYPE_RNN;

public class RNN extends RNNBase {
    // Layer parameters
    public Parameter weight_ih;
    public Parameter weight_hh;
    public Parameter bias_ih;
    public Parameter bias_hh;

    public RNN(int inputSize, int hiddenSize) {
        super(inputSize, hiddenSize, false, false, TYPE_RNN);

        weight_hh = new Parameter(Nd4j.create(hiddenSize, hiddenSize));
        weight_ih = new Parameter(Nd4j.create(hiddenSize, inputSize));
        bias_hh = new Parameter(Nd4j.create(1, hiddenSize));
        bias_ih = new Parameter(Nd4j.create(1, hiddenSize));

        resetParameters();
    }

    protected Tensor[] rnn_impl(Tensor input, Tensor[] outList, Tensor prevHidden, int seqLen, Tensor prev_cell) {
        for (int i = 0; i < seqLen; i++) {
            Tensor currIn = input.get(i);
            Tensor currOut = calculate_gate(currIn, prevHidden, weight_ih, weight_hh, bias_hh, bias_ih, null);

            currOut = compute_current(currOut);

            outList[i] = prevHidden = h_n = currOut;
        }
        return outList;
    }

    public void setParam(INDArray param, int param_type) {
        switch (param_type){
            case WEIGHT_HH:
                weight_hh.data = param;
                break;
            case WEIGHT_IH:
                weight_ih.data = param;
                break;
            case BIAS_HH:
                bias_hh.data = param;
                break;
            case BIAS_IH:
                bias_ih.data = param;
        }
    }

    private void resetParameters() {
        double stdv = 1.0 / Math.sqrt(this.hiddenSize);
        for (Parameter weight : this.parameters())
            if (weight != null)
                WeightInit.uniform(weight.data, -stdv, stdv);
    }

}
