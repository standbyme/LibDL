package LibDL.nn;

import LibDL.Tensor.Variable;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.stream.Stream;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class RNN extends Tensor {

    // Layer configurations
    private long inputSize;
    private long hiddenSize;
    // TODO: fix hard-coding here
    private IActivation activation = new ActivationTanH();

    // Layer parameters
    Variable weight_ih;
    Variable weight_hh;
    Variable bias_ih;
    Variable bias_hh;

    // Input
    Tensor input;
    Variable h0;

    // Output
    private Variable hidden;

    public RNN(int inputSize, int hiddenSize) {
        super();
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        weight_hh = new Variable(Nd4j.create(hiddenSize, hiddenSize), true);
        weight_ih = new Variable(Nd4j.create(inputSize, hiddenSize),true);
        bias_hh = new Variable(Nd4j.create(1, hiddenSize), true);
        bias_ih = new Variable(Nd4j.create(1, hiddenSize), true);
    }

    public void setInput(Tensor input, Variable h0) {
        this.input = input;
        this.h0 = h0;
    }

    public void forward() {
        out = forwardHelper();
    }

    @Override
    public void backward() {
        backwardHelper(dout);
        input.backward();
    }


    @Override
    public Variable[] parameters_core() {
        return Stream.concat(Arrays.stream(input.parameters_core()),
                Arrays.stream(new Variable[]{weight_ih, weight_hh, bias_ih, bias_hh}))
                .toArray(Variable[]::new);
    }


    INDArray forwardHelper() {
        hidden = new Variable(Nd4j.create(input.out.shape()[0], input.out.shape()[1], hiddenSize), true);
        INDArray prevHidden = h0.value;
        long times = input.out.size(0);

        for(long i = 0; i < times; i++) {
            INDArray currIn = input.out.get(point(i), all(), all());
            INDArray currOut =  hidden.value.get(point(i), all(), all());
            currOut.assign(currIn.mmul(weight_ih.value.transpose())
                    .add(prevHidden.mmul(weight_hh.value.transpose()))
                    .addRowVector(bias_hh.value)
                    .addRowVector(bias_ih.value)
            );
            activation.getActivation(currOut, true);
            prevHidden = currOut;
        }

        return hidden.value;
    }

    void backwardHelper(INDArray epsilon) {
        input.dout = Nd4j.emptyLike(input.out);
        weight_hh.dout = Nd4j.zerosLike(weight_hh.value);
        weight_ih.dout = Nd4j.zerosLike(weight_ih.value);
        bias_hh.dout = Nd4j.zerosLike(bias_hh.value);
        bias_ih.dout = Nd4j.zerosLike(bias_ih.value);

        INDArray dzNext = null;
        for(long i = input.out.size(0) - 1; i >= 0; i--) {
            INDArray epsCurrent = epsilon.get(point(i), all(), all());
            INDArray hiddenCurrent = hidden.value.get(point(i), all(), all());
            INDArray inCurrent = input.out.get(point(i), all(), all());
            INDArray hiddenPrevious = (i == 0) ? h0.value : hidden.value.get(point(i - 1), all(), all());
            INDArray epsOutCurrent = input.dout.get(point(i), all(), all());

            if(dzNext != null)
                epsCurrent.addi(dzNext.mmul(weight_hh.value));

            INDArray dzCurrent = epsCurrent.mul(hiddenCurrent.mul(hiddenCurrent).mul(-1).add(1));

            epsOutCurrent.assign(dzCurrent.mmul(weight_ih.value));

            weight_ih.dout.addi(dzCurrent.transpose().mmul(inCurrent));
            weight_hh.dout.addi(dzCurrent.transpose().mmul(hiddenPrevious));
            bias_hh.dout.addi(dzCurrent.sum(0));
            bias_ih.dout.addi(dzCurrent.sum(0));

            dzNext = dzCurrent;
        }
    }

}