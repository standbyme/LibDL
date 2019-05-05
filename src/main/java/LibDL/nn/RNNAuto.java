package LibDL.nn;

import LibDL.Tensor.Variable;
import LibDL.Tensor.Operator.Concat;
import LibDL.Tensor.Operator.Tanh;
import LibDL.Tensor.Tensor;
import LibDL.Tensor.Parameter;
import org.nd4j.linalg.factory.Nd4j;

public class RNNAuto extends Module {

    // Layer configurations
    private long inputSize;
    private long hiddenSize;

    // Layer parameters
    Parameter weight_ih;
    Parameter weight_hh;
    Parameter bias_ih;
    Parameter bias_hh;

    private Variable h0;

    public RNNAuto(int inputSize, int hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        weight_hh = new Parameter(Nd4j.create(hiddenSize, hiddenSize));
        weight_ih = new Parameter(Nd4j.create(inputSize, hiddenSize));
        bias_hh = new Parameter(Nd4j.create(1, hiddenSize));
        bias_ih = new Parameter(Nd4j.create(1, hiddenSize));

        resetParameters();
    }

    @Override
    public Tensor forward(Tensor input) {
        int seqLen = (int)input.size(0);
        int batchSize = (int)input.size(1);
        if(h0 == null)
            h0 = new Variable(Nd4j.zeros(batchSize, hiddenSize));

        Tensor[] outList = new Tensor[seqLen];
        Tensor prevHidden = h0;

        for(int i = 0; i < seqLen; i++) {
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

        return output.reshape(seqLen, batchSize, hiddenSize);
    }

    public void setH0(Variable h0) {
        this.h0 = h0;
    }

    private void resetParameters() {
        double stdv = 1.0 / Math.sqrt(this.hiddenSize);
        for(Parameter weight : this.parameters())
            WeightInit.uniform(weight.data, -stdv, stdv);
    }

    @Override
    public String toString() {
        return "RNN(inputSize=" + inputSize + ", hiddenSize=" + hiddenSize + ")";
    }

}
