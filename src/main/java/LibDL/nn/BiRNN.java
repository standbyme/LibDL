package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.stream.Stream;

import static org.nd4j.linalg.indexing.NDArrayIndex.*;


public class BiRNN extends Tensor {

    // Layer configurations
    private long inputSize;
    private long hiddenSize;

    // Input
    Tensor input;
    Constant h0;

    // Output
    private Constant hidden;

    private RNN forwardRNN;
    private RNN backwardRNN;

    public BiRNN(int inputSize, int hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        forwardRNN = new RNN(inputSize, hiddenSize);
        backwardRNN = new RNN(inputSize, hiddenSize);
    }

    public void setInput(Tensor input, Constant h0) {
        this.input = input;
        this.h0 = h0;
    }

    @Override
    public void forward() {
        input.forward();

        Tensor inputReversed = reverseInput(input);
        Constant h0F = new Constant(h0.value.get(point(0), all(), all()));
        Constant h0B = new Constant(h0.value.get(point(1), all(), all()));

        hidden = new Constant(Nd4j.create(input.out.shape()[0], input.out.shape()[1], hiddenSize*2), true);

        forwardRNN.setInput(input, h0F);
        forwardRNN.forward();

        backwardRNN.setInput(inputReversed, h0B);
        backwardRNN.forward();

        hidden.value.assign(Nd4j.hstack(forwardRNN.out, backwardRNN.out));
    }

    private Constant reverseInput(Tensor input) {
        INDArray rev = Nd4j.emptyLike(input.out);
        long times = input.out.size(0);
        for(long i = 0; i < times; i++) {
            rev.get(point(i), all(), all()).assign(input.out.get(point(times - 1 - i), all(), all()));
        }
        return new Constant(rev);
    }

    @Override
    public void backward() {
        forwardRNN.dout = dout.get(all(), all(), interval(0, hiddenSize));
        backwardRNN.dout = dout.get(all(), all(), interval(hiddenSize+1, hiddenSize*2));

        forwardRNN.backward();
        backwardRNN.backward();
    }

    @Override
    public Constant[] parameters() {
        return Stream.concat(Arrays.stream(forwardRNN.parameters()), Arrays.stream(backwardRNN.parameters()))
                .toArray(Constant[]::new);
    }

}