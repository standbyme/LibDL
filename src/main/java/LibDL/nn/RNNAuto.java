package LibDL.nn;

import LibDL.Tensor.Operator.Concat;
import LibDL.Tensor.Parameter;
import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.nd4j.linalg.factory.Nd4j;

public class RNNAuto extends Module {

    // Layer configurations
    protected long inputSize;
    protected long hiddenSize;


    // Layer parameters
    Parameter weight_ih;
    Parameter weight_hh;
    Parameter bias_ih;
    Parameter bias_hh;

    Parameter gu_weight_ih;
    Parameter gu_weight_hh;
    Parameter gu_bias_ih;
    Parameter gu_bias_hh;

    Parameter gro_weight_ih;
    Parameter gro_weight_hh;
    Parameter gro_bias_ih;
    Parameter gro_bias_hh;

    Parameter gf_weight_ih;
    Parameter gf_weight_hh;
    Parameter gf_bias_ih;
    Parameter gf_bias_hh;

    protected Variable h0;

    protected boolean relu;

    protected boolean batch_first;

//    protected RNNAuto(int inputSize, int hiddenSize, String type) {
//        is_lstm = type.equals("LSTM");
//        is_gru = type.equals("GRU");
//        is_relu = type.equals("RELU");
//    }

    static protected int TYPE_RNN = 0, TYPE_LSTM = 1, TYPE_GRU = 2;

    public RNNAuto(int inputSize, int hiddenSize) {
        this(inputSize, hiddenSize, false, false, TYPE_RNN);
        assert !batch_first;
    }

    public RNNAuto(int inputSize, int hiddenSize, boolean relu, boolean batch_first, int type) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        this.relu = relu;
        this.batch_first = batch_first;

//        weight_hh = new Parameter(Nd4j.create(hiddenSize, hiddenSize));
//        weight_ih = new Parameter(Nd4j.create(inputSize, hiddenSize));
//        bias_hh = new Parameter(Nd4j.create(1, hiddenSize));
//        bias_ih = new Parameter(Nd4j.create(1, hiddenSize));

        init_params(weight_hh, weight_ih, bias_hh, bias_ih);

        if (type == TYPE_GRU) {
            init_params(gu_weight_hh, gu_bias_ih, gu_bias_hh, gu_bias_ih);
            init_params(gro_weight_hh, gro_bias_ih, gro_bias_hh, gro_bias_ih);
        } else if (type == TYPE_LSTM) {
            init_params(gu_weight_hh, gu_bias_ih, gu_bias_hh, gu_bias_ih);
            init_params(gf_weight_hh, gf_bias_ih, gf_bias_hh, gf_bias_ih);
            init_params(gro_weight_hh, gro_bias_ih, gro_bias_hh, gro_bias_ih);
        }
        resetParameters();
    }

    protected void init_params(Parameter... parameters) {
        parameters[0] = new Parameter(Nd4j.create(hiddenSize, hiddenSize));
        parameters[1] = new Parameter(Nd4j.create(inputSize, hiddenSize));
        parameters[2] = new Parameter(Nd4j.create(1, hiddenSize));
        parameters[3] = new Parameter(Nd4j.create(1, hiddenSize));
    }

    protected Tensor calculate_gate(Tensor input, Tensor last,
                                    Tensor w1, Tensor w2,
                                    Tensor b1, Tensor b2,
                                    Tensor r) {
        if (r == null) {
            return input.mm(w1.transpose())
                    .add(last.mm(w2.transpose()))
                    .addVector(b1).addVector(b2);
        } else {
            Tensor rlast = last.mul(r);
            return input.mm(w1.transpose())
                    .add(rlast.mm(w2.transpose()))
                    .addVector(b1).addVector(b2);
        }
    }

    @Override
    public Tensor forward(Tensor... input) {
        this.setH0((Variable) input[1]);
        return forward(input[0]);
    }

    protected Tensor compute_current(Tensor tensor) {
        return relu ? Functional.relu(tensor) : Tensor.tanh(tensor);
    }

    protected void forward_loop(Tensor input, Tensor[] outList, Tensor prevHidden, int seqLen) {

        for (int i = 0; i < seqLen; i++) {
            Tensor currIn = input.get(i);
            Tensor r = null;

            Tensor currOut = calculate_gate(currIn, prevHidden, weight_ih, weight_hh, bias_hh, bias_ih, r);

            currOut = compute_current(currOut);

            outList[i] = currOut;
            prevHidden = currOut;
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        int seqLen = (int) input.size(0);
        int batchSize = (int) input.size(1);
        if (h0 == null)
            h0 = new Variable(Nd4j.zeros(batchSize, hiddenSize));

        Tensor[] outList = new Tensor[seqLen];
        Tensor prevHidden = h0;

        forward_loop(input, outList, prevHidden, seqLen);

        Tensor output = new Concat(outList);

        return output.reshape(seqLen, batchSize, hiddenSize);
    }

    public void setH0(Variable h0) {
        this.h0 = h0;
    }

    private void resetParameters() {
        double stdv = 1.0 / Math.sqrt(this.hiddenSize);
        for (Parameter weight : this.parameters())
            if (weight != null)
                WeightInit.uniform(weight.data, -stdv, stdv);
    }

    @Override
    public String toString() {
        return getClass().getName() +
                "(inputSize=" + inputSize + ", hiddenSize=" + hiddenSize + ")";
    }

}
