package LibDL.nn;

import LibDL.Tensor.Constant;
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
    private Constant weight_ih;
    private Constant weight_hh;
    private Constant bias_ih;
    private Constant bias_hh;

    // Input
    private Tensor input;
    private Constant h0;

    // Output
    private Constant hidden;

    public RNN(int inputSize, int hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        weight_hh = new Constant(Nd4j.create(hiddenSize, hiddenSize), true);
        weight_ih = new Constant(Nd4j.create(inputSize, hiddenSize),true);
        bias_hh = new Constant(Nd4j.create(1, hiddenSize), true);
        bias_ih = new Constant(Nd4j.create(1, hiddenSize), true);
    }

    public void setInput(Tensor input, Constant h0) {
        this.input = input;
        this.h0 = h0;
    }

    @Override
    public void forward() {
        input.forward();
        out = forwardHelper();
    }

    @Override
    public void backward() {
        backwardHelper(dout);
        input.backward();
    }

    @Override
    public Constant[] parameters() {
        return Stream.concat(Arrays.stream(input.parameters()),
                Arrays.stream(new Constant[]{weight_ih, weight_hh, bias_ih, bias_hh}))
                .toArray(Constant[]::new);
    }

    INDArray forwardHelper() {
        hidden = new Constant(Nd4j.create(input.out.shape()[0], input.out.shape()[1], hiddenSize), true);
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
    
    public static void main(String[] args) {
        RNN rnn = new RNN(3, 5);
        rnn.weight_hh = new Constant(Nd4j.create(new double[][]{
                {-0.2759, -0.2183,  0.4454, -0.0331,  0.0015},
                {-0.4196, -0.1691, -0.1807,  0.3347, -0.2393},
                {-0.3233,  0.4190, -0.3819, -0.1739, -0.2363},
                {-0.0749,  0.1831,  0.1638,  0.1701,  0.4200},
                { 0.4375, -0.1900, -0.3810, -0.2224, -0.4320}}));

        rnn.weight_ih = new Constant(Nd4j.create(new double[][]{
                {-0.0862,  0.1885,  0.1464},
                {-0.0782, -0.0145, -0.2172},
                { 0.1067,  0.0100, -0.4008},
                {-0.0921, -0.1040,  0.1249},
                { 0.0167, -0.1631,  0.1717}}));

        rnn.bias_hh = new Constant(Nd4j.create(new double[][]{{-0.1757, -0.2823, -0.3362, -0.1846, -0.0046}}));

        rnn.bias_ih = new Constant(Nd4j.create(new double[][]{{-0.1291, -0.2665, -0.0902,  0.3374,  0.2181}}));

        INDArray input = Nd4j.create(new double[][][] {
                {{ 0.0053, -0.4601, -0.5414},
                        {-0.8522,  0.5896,  0.5357},
                        {-1.4926,  1.1163, -0.1855}},

                {{0.2956,  1.4636,  0.6051},
                        {1.0905,  0.5851, -0.4907},
                        {1.0333, -0.2276, -0.1532}}}
        );

        INDArray h0 = Nd4j.create(new double[][] {
                { 0.4765,  0.3493, -0.8491,  0.7070,  0.6665},
                {-0.3468, -0.0986, -0.5787, -0.6640,  0.5776},
                {-0.6707,  0.6514,  0.6557,  0.1384, -0.9628}});

        rnn.setInput(new Constant(input), new Constant(h0));
        rnn.forward();
        INDArray output = rnn.out;

        System.out.println(output);

        INDArray epsilon = Nd4j.create(new double[][][] {{
                { 1.1837e+00,  2.8680e-02,  5.9473e-01, -6.3787e-01, -9.8196e-01},
                {-1.5350e+00,  7.6189e-01,  1.3230e+00, -4.2295e-01,  5.3208e-01},
                {-1.4129e+00, -2.3161e+00, -2.5905e-02,  1.8038e+00, -7.0832e-01}},

                {{-1.7072e+00, -3.1917e+00,  1.1541e+00, -1.9135e+00,  2.3066e-01},
                        {-3.6231e-03,  3.9820e-01,  4.9735e-01, -1.5231e+00, -2.8920e-03},
                        {-4.8295e-01, -2.3305e+00, -1.2397e+00,  1.6851e+00,  1.8875e-01}}}
        );

        rnn.backwardHelper(epsilon);
        rnn.parameters();
    }


}