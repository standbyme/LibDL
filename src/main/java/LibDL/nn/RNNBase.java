package LibDL.nn;

import LibDL.Tensor.Operator.Concat;
import LibDL.Tensor.Parameter;
import LibDL.Tensor.Tensor;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;


abstract public class RNNBase extends Module {
    public static final int WEIGHT_IH = 0, WEIGHT_HH = 1, BIAS_IH = 2, BIAS_HH = 3;

    // Layer configurations
    protected int inputSize;
    protected int hiddenSize;
    protected int numLayers;
    protected int numDirections;

    //use ReLU instead of tanh
    protected boolean relu;

    //dropout
    protected boolean dropout;
    protected double dropout_p;

    //use bias
    protected boolean bias;

    //not implemented
    protected boolean batch_first;
    protected boolean bidirectional;



    protected int PARAM_I = 0, PARAM_F = 1, PARAM_G = 2, PARAM_O = 3;
    protected int PARAM_R = 0, PARAM_Z = 1, PARAM_N = 2;

    public Parameter[] weight_ih;
    public Parameter[] weight_hh;
    public Parameter[] bias_ih;
    public Parameter[] bias_hh;

    protected RNNType rnn_type;

    protected int pm(int layer, int param_type) {
        return rnn_type.gateSize() * layer + param_type;
    }

    protected void init_param() {
        weight_hh = new Parameter[rnn_type.gateSize() * numLayers];
        weight_ih = new Parameter[rnn_type.gateSize() * numLayers];
        if (bias) {
            bias_hh = new Parameter[rnn_type.gateSize() * numLayers];
            bias_ih = new Parameter[rnn_type.gateSize() * numLayers];
        }
        double stdv = 1.0 / Math.sqrt(this.hiddenSize);
        for (int layer = 0; layer < numLayers; layer++) {
            for (int i = 0; i < rnn_type.gateSize(); i++) {
                weight_hh[pm(layer, i)] = new Parameter(Nd4j.create(hiddenSize, hiddenSize));
                weight_ih[pm(layer, i)] = new Parameter(Nd4j.create(hiddenSize, layer == 0 ? inputSize : numDirections * hiddenSize));
                if (bias) {
                    bias_hh[pm(layer, i)] = new Parameter(Nd4j.create(1, hiddenSize));
                    bias_ih[pm(layer, i)] = new Parameter(Nd4j.create(1, hiddenSize));
                }
            }
            for (int i = 0; i < rnn_type.gateSize(); i++) {
                WeightInit.uniform(weight_hh[pm(layer, i)].data, -stdv, stdv);
                WeightInit.uniform(weight_ih[pm(layer, i)].data, -stdv, stdv);
                if (bias) {
                    WeightInit.uniform(bias_hh[pm(layer, i)].data, -stdv, stdv);
                    WeightInit.uniform(bias_ih[pm(layer, i)].data, -stdv, stdv);
                }
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
        this.bidirectional = bidirectional;
        this.numDirections = bidirectional ? 2 : 1;

        this.dropout = dropout != 0;
        this.dropout_p = dropout;

        this.h_n = new Tensor[numLayers];
        init_param();
    }

    @Override
    public Tensor forward(Tensor t) {
        throw new UnsupportedOperationException("Please call forward(inut, h0)");
    }

    @Override
    public Tensor forward(Tensor... tensors) {
        if (this instanceof LSTM) {
            return forward_check(tensors[0], tensors[1], tensors[2]);
        } else return forward_check(tensors[0], tensors[1]);
    }

    private void check_input_size(long[] size, long[] target) {
        if (!Arrays.equals(size, target)) throw new RuntimeException("Input size mismatch, excepted "
                + Arrays.toString(target)
                + ", got "
                + Arrays.toString(size));

    }

    private Tensor forward_check(@NotNull Tensor input, @NotNull Tensor h0) {
        check_input_size(h0.sizes(), new long[]{numLayers * numDirections, input.size(batch_first ? 0 : 1), hiddenSize});
        return forward_(input, h0, null);
    }

    private Tensor forward_check(@NotNull Tensor input, @NotNull Tensor h0, @NotNull Tensor c0) {
        long[] sizes = new long[]{numLayers * numDirections, input.size(batch_first ? 0 : 1), hiddenSize};
        check_input_size(h0.sizes(), sizes);
        check_input_size(c0.sizes(), sizes);
        return forward_(input, h0, c0);
    }


    protected Tensor forward_(Tensor input, Tensor h0, Tensor c0) {
        int seqLen = (int) input.size(0);
        int batchSize = (int) input.size(1);

        Tensor[] outList = new Tensor[seqLen];
        Tensor output = input;
        Tensor nxt = output;

        for (int layer = 0; layer < numLayers; layer++) {
            Tensor prevHidden = h0.get(layer);
            Tensor prevC = c0;
            if (c0 != null) prevC = c0.get(layer);
            outList = rnn_impl(nxt, outList, prevHidden, seqLen, prevC, layer);
            nxt = output = new Concat(0, outList).reshape(seqLen, batchSize, hiddenSize);
            if (dropout) nxt = Functional.dropout(output, dropout_p);

        }

        return output;
    }

    protected abstract Tensor[] rnn_impl(Tensor input, Tensor[] outList,
                                         Tensor prevHidden, int seqLen,
                                         Tensor prev_cell, int currLayer);

    protected Tensor calculate_gate(@NotNull Tensor input, @NotNull Tensor last,
                                    @NotNull Tensor w_ih, @NotNull Tensor w_hh,
                                    @Nullable Tensor b_hh, @Nullable Tensor b_ih,
                                    @Nullable Tensor r_gate) {
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

    public Tensor hn() {
        return new Concat(0, h_n);
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
