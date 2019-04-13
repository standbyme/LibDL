package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Operator.Concat;
import LibDL.Tensor.Operator.Tanh;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.factory.Nd4j;

public class RNNAuto extends LayerTensor {

    // Layer configurations
    private long inputSize;
    private long hiddenSize;

    // Layer parameters
    Constant weight_ih;
    Constant weight_hh;
    Constant bias_ih;
    Constant bias_hh;

    private Constant h0;

    @Override
    protected Tensor core() {
        int times = (int)input.size(0);

        Tensor[] outList = new Tensor[times];
        Tensor prevHidden = h0;

        for(int i = 0; i < times; i++) {
            Tensor currIn = input.get(i);
            Tensor currOut = currIn.mm(weight_ih.transpose())
                    .add(prevHidden.mm(weight_hh.transpose()))
                    .addVector(bias_hh)
                    .addVector(bias_ih);
            currOut = new Tanh(currOut);
            outList[i] = currOut;
            prevHidden = currOut;
        }

        Tensor output = new Concat(outList);

        return output.reshape(times, input.size(1), hiddenSize);
    }

    public RNNAuto(int inputSize, int hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        weight_hh = new Constant(Nd4j.create(hiddenSize, hiddenSize), true);
        weight_ih = new Constant(Nd4j.create(inputSize, hiddenSize),true);
        bias_hh = new Constant(Nd4j.create(1, hiddenSize), true);
        bias_ih = new Constant(Nd4j.create(1, hiddenSize), true);
    }

    public void setH0(Constant h0) {
        this.h0 = h0;
    }
}
