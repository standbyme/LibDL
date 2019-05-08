package LibDL.nn;

import LibDL.Tensor.Operator.Concat;
import LibDL.Tensor.Parameter;
import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class RNNAuto extends Module {

    // Layer configurations
    protected long inputSize;
    protected long hiddenSize;

    // Layer parameters
    public Parameter weight_ih;
    public Parameter weight_hh;
    public Parameter bias_ih;
    public Parameter bias_hh;

    protected Variable h0;
    protected Variable c0;

    public Tensor h_n;

    //use ReLU instead of tanh
    protected boolean relu;

    //not implemented
    protected boolean batch_first;

    protected int rnn_type;

    public static final int TYPE_RNN = 1, TYPE_LSTM = 4, TYPE_GRU = 3,
            WEIGHT_IH = 0, WEIGHT_HH = 1, BIAS_IH = 2, BIAS_HH = 3,
            PARAM = 0, OUTPUT = 1, RESET = 1, UPDATE = 2, FORGET = 3;


    public RNNAuto(int inputSize, int hiddenSize) {
        this(inputSize, hiddenSize, false, false, TYPE_RNN);
    }

    public RNNAuto(int inputSize, int hiddenSize, boolean relu, boolean batch_first, int type) {
        assert !batch_first;//batch_first not implemented

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        this.relu = relu;
        this.batch_first = batch_first;
        this.rnn_type = type;

        weight_hh = new Parameter(Nd4j.create(hiddenSize, hiddenSize));
        weight_ih = new Parameter(Nd4j.create(hiddenSize, inputSize));
        bias_hh = new Parameter(Nd4j.create(1, hiddenSize));
        bias_ih = new Parameter(Nd4j.create(1, hiddenSize));

        resetParameters();
    }

    public void setParam(INDArray param, int param_type) {
        INDArrayIndex[] indices = new INDArrayIndex[param.rank()];
        for (int i = 1; i < indices.length; i++) {
            indices[i] = NDArrayIndex.all();
        }

        for (int i = 0; i < 4; i++) {
            indices[0] = NDArrayIndex.interval(i * hiddenSize, i * hiddenSize + hiddenSize);
            param.get(indices);
        }
    }


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

    @Override
    public Tensor forward(Tensor... input) {
        this.setH0((Variable) input[1]);
        if (rnn_type == TYPE_LSTM) {
            this.setC0((Variable) input[2]);
        }
        return forward(input[0]);
    }

    protected Tensor compute_current(Tensor tensor) {
        return relu ? Functional.relu(tensor) : Tensor.tanh(tensor);
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

    @Override
    public Tensor forward(Tensor input) {
        int seqLen = (int) input.size(0);
        int batchSize = (int) input.size(1);
        if (h0 == null)
            h0 = new Variable(Nd4j.zeros(batchSize, hiddenSize));

        Tensor[] outList = new Tensor[seqLen];
        Tensor prevHidden = h0;


        outList = rnn_impl(input, outList, prevHidden, seqLen, c0);

        Tensor output = new Concat(outList);

        return output.reshape(seqLen, batchSize, hiddenSize);
    }

    public void setH0(Variable h0) {
        this.h0 = h0;
    }

    public void setC0(Variable c0) {
        this.c0 = c0;
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
