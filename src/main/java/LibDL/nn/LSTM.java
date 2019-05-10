package LibDL.nn;

import LibDL.Tensor.Parameter;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static LibDL.nn.RNNBase.RNNType.TYPE_LSTM;

public class LSTM extends RNNBase {
    // Layer parameters
    private Parameter weight_ih;
    private Parameter weight_hh;
    private Parameter bias_ih;
    private Parameter bias_hh;

    //update gate params
    private Parameter gu_weight_ih;
    private Parameter gu_weight_hh;
    private Parameter gu_bias_ih;
    private Parameter gu_bias_hh;

    //reset(GRU)/output(LSTM) gate params
    private Parameter gro_weight_ih;
    private Parameter gro_weight_hh;
    private Parameter gro_bias_ih;
    private Parameter gro_bias_hh;

    //forget(LSTM) gate params
    private Parameter gf_weight_ih;
    private Parameter gf_weight_hh;
    private Parameter gf_bias_ih;
    private Parameter gf_bias_hh;

    public Tensor c_n;

    public LSTM(int inputSize, int hiddenSize) {
        super(inputSize, hiddenSize, false, false, TYPE_LSTM);
        weight_hh = new Parameter(Nd4j.create(hiddenSize, hiddenSize));
        weight_ih = new Parameter(Nd4j.create(hiddenSize, inputSize));
        bias_hh = new Parameter(Nd4j.create(1, hiddenSize));
        bias_ih = new Parameter(Nd4j.create(1, hiddenSize));

        gu_weight_hh = new Parameter(Nd4j.create(hiddenSize, hiddenSize));
        gu_weight_ih = new Parameter(Nd4j.create(hiddenSize, inputSize));
        gu_bias_hh = new Parameter(Nd4j.create(1, hiddenSize));
        gu_bias_ih = new Parameter(Nd4j.create(1, hiddenSize));

        gro_weight_hh = new Parameter(Nd4j.create(hiddenSize, hiddenSize));
        gro_weight_ih = new Parameter(Nd4j.create(hiddenSize, inputSize));
        gro_bias_hh = new Parameter(Nd4j.create(1, hiddenSize));
        gro_bias_ih = new Parameter(Nd4j.create(1, hiddenSize));

        gf_weight_hh = new Parameter(Nd4j.create(hiddenSize, hiddenSize));
        gf_weight_ih = new Parameter(Nd4j.create(hiddenSize, inputSize));
        gf_bias_hh = new Parameter(Nd4j.create(1, hiddenSize));
        gf_bias_ih = new Parameter(Nd4j.create(1, hiddenSize));
    }

    @Override
    protected Tensor[] rnn_impl(Tensor input, Tensor[] outList, Tensor prevHidden, int seqLen, Tensor prev_cell) {
        for (int i = 0; i < seqLen; i++) {
            Tensor currIn = input.get(i);
            Tensor g_t = calculate_gate(currIn, prevHidden, weight_ih, weight_hh, bias_hh, bias_ih, null);

            g_t = compute_current(g_t);

            Tensor i_t = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, gu_weight_ih, gu_weight_hh, gu_bias_hh, gu_bias_ih, null)
            );
            Tensor f_t = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, gf_weight_ih, gf_weight_hh, gf_bias_hh, gf_bias_ih, null)
            );
            Tensor o_t = Functional.sigmoid(
                    calculate_gate(currIn, prevHidden, gro_weight_ih, gro_weight_hh, gro_bias_hh, gro_bias_ih, null)
            );

            Tensor new_cell = f_t.mul(prev_cell).add(i_t.mul(g_t));

            h_n = prevHidden = outList[i] = o_t.mul(Tensor.tanh(new_cell));
            c_n = prev_cell = new_cell;
        }
        return outList;
    }

    public void setParam(INDArray param, int param_type) {
        Parameter[] paramList = null;
        switch (param_type){
            case WEIGHT_HH:
                paramList = new Parameter[]{gu_weight_hh, gf_weight_hh, weight_hh, gro_weight_hh};
                break;
            case WEIGHT_IH:
                paramList = new Parameter[]{gu_weight_ih, gf_weight_ih, weight_ih, gro_weight_ih};
                break;
            case BIAS_HH:
                paramList = new Parameter[]{gu_bias_hh, gf_bias_hh, bias_hh, gro_bias_hh};
                break;
            case BIAS_IH:
                paramList = new Parameter[]{gu_bias_ih, gf_bias_ih, bias_ih, gro_bias_ih};
        }

        INDArrayIndex[] indices = new INDArrayIndex[param.rank()];
        for (int i = 1; i < indices.length; i++) {
            indices[i] = NDArrayIndex.all();
        }

        for (int i = 0; i < 4; i++) {
            indices[0] = NDArrayIndex.interval(i * hiddenSize, i * hiddenSize + hiddenSize);
            assert paramList != null;
            paramList[i].data = param.get(indices);
        }
    }


}
