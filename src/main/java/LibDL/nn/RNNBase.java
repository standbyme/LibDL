package LibDL.nn;

import LibDL.Tensor.Operator.Concat;
import LibDL.Tensor.Parameter;
import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;


abstract public class RNNBase extends Module {
    public static final int WEIGHT_IH = 0, WEIGHT_HH = 1, BIAS_IH = 2, BIAS_HH = 3;

    // Layer configurations
    protected long inputSize;
    protected long hiddenSize;
    protected long numLayers;

    //use ReLU instead of tanh
    protected boolean relu;

    //dropout
    protected boolean dropout;
    protected double dropout_p;

    //use bias
    protected boolean bias;

    //not implemented
    protected boolean batch_first;
    protected boolean bid;

    public enum RNNType {
        TYPE_RNN(1),
        TYPE_LSTM(4),
        TYPE_GRU(3);

        private final int value;

        RNNType(int value) {
            this.value = value;
        }

        public int gateSize() {
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
        weight_hh = new Parameter[rnn_type.gateSize()];
        weight_ih = new Parameter[rnn_type.gateSize()];
        if (bias) {
            bias_hh = new Parameter[rnn_type.gateSize()];
            bias_ih = new Parameter[rnn_type.gateSize()];
        }
        double stdv = 1.0 / Math.sqrt(this.hiddenSize);
        for (int i = 0; i < rnn_type.gateSize(); i++) {
            weight_hh[i] = new Parameter(Nd4j.create(hiddenSize, hiddenSize));
            weight_ih[i] = new Parameter(Nd4j.create(hiddenSize, inputSize));
            if (bias) {
                bias_hh[i] = new Parameter(Nd4j.create(1, hiddenSize));
                bias_ih[i] = new Parameter(Nd4j.create(1, hiddenSize));
            }
        }
        for (int i = 0; i < rnn_type.gateSize(); i++) {
            WeightInit.uniform(weight_hh[i].data, -stdv, stdv);
            WeightInit.uniform(weight_ih[i].data, -stdv, stdv);
            if (bias) {
                WeightInit.uniform(bias_hh[i].data, -stdv, stdv);
                WeightInit.uniform(bias_ih[i].data, -stdv, stdv);
            }
        }
    }

    public Tensor[] h_n;

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
        this.bias = bias;

        this.dropout = dropout != 0;
        this.dropout_p = dropout;

        this.h_n = new Tensor[numLayers];
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
        Tensor nxt = output;
        Tensor prevHidden = h0;

        for (int layer = 0; layer < numLayers; layer++) {
            outList = rnn_impl(nxt, outList, prevHidden, seqLen, c0, layer);
            output = new Concat(0, outList).reshape(seqLen, batchSize, hiddenSize);
            if (dropout) nxt = Functional.dropout(output, dropout_p);

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

        Tensor rlast = last.mm(w_hh.transpose());
        if (bias) rlast = rlast.addVector(b_hh);
        if (r_gate != null) rlast = rlast.mul(r_gate);
        Tensor result = input.mm(w_ih.transpose()).add(rlast);
        if (bias) result = result.addVector(b_ih);
        return result;
    }

    //A shortcut
    protected Tensor calculate_gate(Tensor input, Tensor last,
                                    int currLayer, int param_type,
                                    Tensor r_gate) {
        return calculate_gate(input, last,
                weight_ih[pm(currLayer, param_type)],
                weight_hh[pm(currLayer, param_type)],
                bias_hh[pm(currLayer, param_type)],
                bias_ih[pm(currLayer, param_type)], r_gate);
    }

    protected Tensor compute_current(Tensor tensor) {
        return relu ? Functional.relu(tensor) : Tensor.tanh(tensor);
    }

    @Override
    public String toString() {
        return getClass().getName() +
                "(inputSize=" + inputSize + ", hiddenSize=" + hiddenSize + ")";
    }


    public void setParam(int param_type, INDArray... params) {
        Parameter[] paramList = null;
        for (int currLayer = 0; currLayer < numLayers; currLayer++) {
            INDArray param = params[currLayer];
            switch (param_type) {
                case WEIGHT_HH:
                    paramList = weight_hh;
                    break;
                case WEIGHT_IH:
                    paramList = weight_ih;
                    break;
                case BIAS_HH:
                    paramList = bias_hh;
                    break;
                case BIAS_IH:
                    paramList = bias_ih;
            }


            INDArrayIndex[] indices = new INDArrayIndex[param.rank()];
            for (int i = 1; i < indices.length; i++) {
                indices[i] = NDArrayIndex.all();
            }

            for (int gate_type = 0; gate_type < rnn_type.gateSize(); gate_type++) {
                indices[0] = NDArrayIndex.interval(gate_type * hiddenSize, gate_type * hiddenSize + hiddenSize);
                assert paramList != null;
                paramList[pm(currLayer, gate_type)].data = param.get(indices);
            }
        }
    }


}
