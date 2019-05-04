package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.indexing.NDArrayIndex.*;


public class BiRNN extends Tensor {

    // Layer configurations
    private long inputSize;
    private long hiddenSize;

    // Input
    Tensor input;
    Variable h0;

    // Output
    private Variable hidden;

    private RNN forwardRNN;
    private RNN backwardRNN;

    public BiRNN(int inputSize, int hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        forwardRNN = new RNN(inputSize, hiddenSize);
        backwardRNN = new RNN(inputSize, hiddenSize);
    }

    public void setInput(Tensor input, Variable h0) {
        this.input = input;
        this.h0 = h0;
    }

    private void forward() {
        Tensor inputReversed = reverseInput(input);
        Constant h0F = new Constant(h0.data.get(point(0), all(), all()));
        Constant h0B = new Constant(h0.data.get(point(1), all(), all()));

        hidden = new Variable(Nd4j.create(input.data.shape()[0], input.data.shape()[1], hiddenSize * 2), true);

        forwardRNN.setInput(input, h0F);
        forwardRNN.forward();

        backwardRNN.setInput(inputReversed, h0B);
        backwardRNN.forward();

        hidden.data.assign(Nd4j.hstack(forwardRNN.data, backwardRNN.data));
    }

    private Constant reverseInput(Tensor input) {
        INDArray rev = Nd4j.emptyLike(input.data);
        long times = input.data.size(0);
        for (long i = 0; i < times; i++) {
            rev.get(point(i), all(), all()).assign(input.data.get(point(times - 1 - i), all(), all()));
        }
        return new Constant(rev);
    }

    @Override
    public void backward() {
        forwardRNN.grad = grad.get(all(), all(), interval(0, hiddenSize));
        backwardRNN.grad = grad.get(all(), all(), interval(hiddenSize + 1, hiddenSize * 2));

        forwardRNN.backward();
        backwardRNN.backward();
    }

}