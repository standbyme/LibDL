package LibDL.nn;

import LibDL.Tensor.Operator.Concat;
import LibDL.Tensor.Parameter;
import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.nd4j.linalg.factory.Nd4j;


abstract public class RNNBase extends Module {
    public static final int WEIGHT_IH = 0, WEIGHT_HH = 1, BIAS_IH = 2, BIAS_HH = 3;

    // Layer configurations
    protected long inputSize;
    protected long hiddenSize;
    protected long numLayers;

    //use ReLU instead of tanh
    protected boolean relu;
    protected boolean bias;
    protected boolean bid;

    //not implemented
    protected boolean batch_first;

    public enum RNNType {
        TYPE_RNN(1),
        TYPE_LSTM(4),
        TYPE_GRU(3);

        private final int value;

        RNNType(int value) {
            this.value = value;
        }

        public int getValue() {
            return value;
        }
    }

    protected int PARAM_I = 0, PARAM_F = 1, PARAM_G = 2, PARAM_O = 3;
    protected int PARAM_R = 0, PARAM_Z = 1, PARAM_N = 2;

    public Parameter[] weight_ih;
    public Parameter[] weight_hh;
    public Parameter[] bias_ih;
    public Parameter[] bias_hh;

    protected RNNType rnn_type;

    protected int pm(int layer, int param_type) {
        return rnn_type.value * layer + param_type;
    }

    protected void init_param() {
        weight_hh = new Parameter[rnn_type.getValue()];
        weight_ih = new Parameter[rnn_type.getValue()];
        bias_hh = new Parameter[rnn_type.getValue()];
        bias_ih = new Parameter[rnn_type.getValue()];
        double stdv = 1.0 / Math.sqrt(this.hiddenSize);
        for (int i = 0; i < rnn_type.getValue(); i++) {
            weight_hh[i] = new Parameter(Nd4j.create(hiddenSize, hiddenSize));
            weight_ih[i] = new Parameter(Nd4j.create(hiddenSize, inputSize));
            bias_hh[i] = new Parameter(Nd4j.create(1, hiddenSize));
            bias_ih[i] = new Parameter(Nd4j.create(1, hiddenSize));
        }
        for (int i = 0; i < rnn_type.getValue(); i++) {
            WeightInit.uniform(weight_hh[i].data, -stdv, stdv);
            WeightInit.uniform(weight_ih[i].data, -stdv, stdv);
            WeightInit.uniform(bias_hh[i].data, -stdv, stdv);
            WeightInit.uniform(bias_ih[i].data, -stdv, stdv);
        }
    }

    public Tensor h_n;

    public RNNBase(int inputSize, int hiddenSize,
                   int numLayers, boolean bias,
                   boolean relu, boolean batch_first,
                   double dropout, boolean bidirectional,
                   RNNType type) {
        assert !batch_first;//batch_first not implemented

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.numLayers = numLayers;

        this.relu = relu;
        this.batch_first = batch_first;
        this.rnn_type = type;
        init_param();
    }

    public Tensor forward(Tensor input) {
        return forward(input, new Variable(Nd4j.zeros(input.size(1), hiddenSize)), null);
    }

    public Tensor forward(Tensor input, Tensor h0) {
        return forward(input, h0, null);
    }

    public Tensor forward(Tensor input, Tensor h0, Tensor c0) {
        int seqLen = (int) input.size(0);
        int batchSize = (int) input.size(1);
        if (h0 == null)
            h0 = new Variable(Nd4j.zeros(batchSize, hiddenSize));

        Tensor[] outList = new Tensor[seqLen];
        Tensor output = input;
        Tensor prevHidden = h0;

        for (int layer = 0; layer < numLayers; layer++) {
            outList = rnn_impl(output, outList, prevHidden, seqLen, c0, layer);
            output = new Concat(outList).reshape(seqLen, batchSize, hiddenSize);
        }

        return output;
    }

    protected abstract Tensor[] rnn_impl(Tensor input, Tensor[] outList,
                                         Tensor prevHidden, int seqLen,
                                         Tensor prev_cell, int currLayer);

    protected Tensor calculate_gate(Tensor input, Tensor last,
                                    Tensor w_ih, Tensor w_hh,
                                    Tensor b_hh, Tensor b_ih,
                                    Tensor r_gate) {
        //r_gate is only for GRU
        if (r_gate == null) {
            return input.mm(w_ih.transpose())
                    .add(last.mm(w_hh.transpose()))
                    .addVector(b_hh).addVector(b_ih);
        } else {
            Tensor rlast = last.mm(w_hh.transpose()).addVector(b_hh).mul(r_gate);
            return input.mm(w_ih.transpose())
                    .add(rlast)
                    .addVector(b_ih);
        }
    }


    protected Tensor compute_current(Tensor tensor) {
        return relu ? Functional.relu(tensor) : Tensor.tanh(tensor);
    }

    @Override
    public String toString() {
        return getClass().getName() +
                "(inputSize=" + inputSize + ", hiddenSize=" + hiddenSize + ")";
    }

}
