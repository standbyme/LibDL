package LibDL.nn;

import LibDL.Tensor.Operator.Concat;
import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.nd4j.linalg.factory.Nd4j;


abstract public class RNNBase extends Module {
    public static final int WEIGHT_IH = 0, WEIGHT_HH = 1, BIAS_IH = 2, BIAS_HH = 3;

    // Layer configurations
    protected long inputSize;
    protected long hiddenSize;

    //use ReLU instead of tanh
    protected boolean relu;

    //not implemented
    protected boolean batch_first;

    public enum RNNType {TYPE_RNN, TYPE_LSTM, TYPE_GRU}
    protected RNNType rnn_type;

    public Tensor h_n;

    public RNNBase(int inputSize, int hiddenSize, boolean relu, boolean batch_first, RNNType type) {
        assert !batch_first;//batch_first not implemented

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        this.relu = relu;
        this.batch_first = batch_first;
        this.rnn_type = type;
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
        Tensor prevHidden = h0;


        outList = rnn_impl(input, outList, prevHidden, seqLen, c0);

        Tensor output = new Concat(0, outList);

        return output.reshape(seqLen, batchSize, hiddenSize);
    }

    protected abstract Tensor[] rnn_impl(Tensor input, Tensor[] outList, Tensor prevHidden, int seqLen,
                                         Tensor prev_cell);

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
