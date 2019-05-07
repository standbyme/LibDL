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
    Parameter weight_ih;
    Parameter weight_hh;
    Parameter bias_ih;
    Parameter bias_hh;

    //update gate params
    Parameter gu_weight_ih;
    Parameter gu_weight_hh;
    Parameter gu_bias_ih;
    Parameter gu_bias_hh;

    //reset(GRU)/output(LSTM) gate params
    Parameter gro_weight_ih;
    Parameter gro_weight_hh;
    Parameter gro_bias_ih;
    Parameter gro_bias_hh;

    //forget(LSTM) gate params
    Parameter gf_weight_ih;
    Parameter gf_weight_hh;
    Parameter gf_bias_ih;
    Parameter gf_bias_hh;

    Parameter[] real_parameters;
    protected Variable h0;

    //use ReLU instead of tanh
    protected boolean relu;

    //not implemented
    protected boolean batch_first;

    protected int rnn_type;

    static public int TYPE_RNN = 1, TYPE_LSTM = 4, TYPE_GRU = 3,
            WEIGHT_IH = 0, WEIGHT_HH = 1, BIAS_IH = 2, BIAS_HH = 3,
            PARAM = 0, OUTPUT = 1, RESET = 1, UPDATE = 2, FORGET = 3;

    static protected int pos(int gate_type, int param_type) {
        return gate_type * 4 + param_type;
    }

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

        this.real_parameters = new_params(type);
        update_references();
        resetParameters();
    }


    public void setParam(INDArray param, int param_type) {
        INDArrayIndex[] indices = new INDArrayIndex[param.rank()];
        for (int i = 1; i < indices.length; i++) {
            indices[i] = NDArrayIndex.all();
        }
        for (int i = 0; i < rnn_type; i++) {
            indices[0] = NDArrayIndex.interval(i * hiddenSize, i * hiddenSize + hiddenSize);
//            System.out.println("Expected" + Arrays.toString(real_parameters[pos(i, param_type)].sizes()));
//            System.out.println("Gets" + Arrays.toString(param.get(indices).shape()));
            real_parameters[pos(i, param_type)].data = param.get(indices);
        }
        update_references();
    }

    private void update_references() {
        weight_hh = real_parameters[pos(PARAM, WEIGHT_HH)];
        weight_ih = real_parameters[pos(PARAM, WEIGHT_IH)];
        bias_hh = real_parameters[pos(PARAM, BIAS_HH)];
        bias_ih = real_parameters[pos(PARAM, BIAS_IH)];

        if (rnn_type == TYPE_GRU) {
            gu_weight_hh = real_parameters[pos(UPDATE, WEIGHT_HH)];
            gu_weight_ih = real_parameters[pos(UPDATE, WEIGHT_IH)];
            gu_bias_hh = real_parameters[pos(UPDATE, BIAS_HH)];
            gu_bias_ih = real_parameters[pos(UPDATE, BIAS_IH)];

            gro_weight_hh = real_parameters[pos(RESET, WEIGHT_HH)];
            gro_weight_ih = real_parameters[pos(RESET, WEIGHT_IH)];
            gro_bias_hh = real_parameters[pos(RESET, BIAS_HH)];
            gro_bias_ih = real_parameters[pos(RESET, BIAS_IH)];

        } else if (rnn_type == TYPE_LSTM) {
            gu_weight_hh = real_parameters[pos(UPDATE, WEIGHT_HH)];
            gu_weight_ih = real_parameters[pos(UPDATE, WEIGHT_IH)];
            gu_bias_hh = real_parameters[pos(UPDATE, BIAS_HH)];
            gu_bias_ih = real_parameters[pos(UPDATE, BIAS_IH)];

            gro_weight_hh = real_parameters[pos(OUTPUT, WEIGHT_HH)];
            gro_weight_ih = real_parameters[pos(OUTPUT, WEIGHT_IH)];
            gro_bias_hh = real_parameters[pos(OUTPUT, BIAS_HH)];
            gro_bias_ih = real_parameters[pos(OUTPUT, BIAS_IH)];

            gf_weight_hh = real_parameters[pos(FORGET, WEIGHT_HH)];
            gf_weight_ih = real_parameters[pos(FORGET, WEIGHT_IH)];
            gf_bias_hh = real_parameters[pos(FORGET, BIAS_HH)];
            gf_bias_ih = real_parameters[pos(FORGET, BIAS_IH)];
        }
    }

    protected Parameter[] new_params(int gate_count) {
        Parameter[] parameters = new Parameter[4 * gate_count];
        for (int i = 0; i < gate_count; i++) {
            parameters[i * 4 + 0] = new Parameter(Nd4j.create(hiddenSize, inputSize));
            parameters[i * 4 + 1] = new Parameter(Nd4j.create(hiddenSize, hiddenSize));
            parameters[i * 4 + 2] = new Parameter(Nd4j.create(1, hiddenSize));
            parameters[i * 4 + 3] = new Parameter(Nd4j.create(1, hiddenSize));
        }
        return parameters;
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
            Tensor rlast = last.mul(r_gate);
            return input.mm(w_ih.transpose())
                    .add(rlast.mm(w_hh.transpose()))
                    .addVector(b_hh).addVector(b_ih);
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

    protected Tensor[] rnn_impl(Tensor input, Tensor[] outList, Tensor prevHidden, int seqLen) {

        for (int i = 0; i < seqLen; i++) {
            Tensor currIn = input.get(i);
            Tensor currOut = calculate_gate(currIn, prevHidden, weight_ih, weight_hh, bias_hh, bias_ih, null);

            currOut = compute_current(currOut);

            outList[i] = currOut;
            prevHidden = currOut;
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

        outList = rnn_impl(input, outList, prevHidden, seqLen);

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
