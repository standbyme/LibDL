package LibDL.nn;

import LibDL.Tensor.*;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.Supplier;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class RNN extends Module {
    // Layer configurations
    private long inputSize;
    private long hiddenSize;

    // Layer parameters
    Parameter weight_ih;
    Parameter weight_hh;
    Parameter bias_ih;
    Parameter bias_hh;

    public RNN(int inputSize, int hiddenSize) {
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
        System.err.println("Warning: Initial hidden state is not provided. Defaults to zero.");
        return new RNNImpl(input, new Variable(Nd4j.zeros(input.size(1), hiddenSize)));
    }

    public Tensor forward(Tensor input, Tensor h0) {
        return new RNNImpl(input, h0);
    }

    private void resetParameters() {
        double stdv = 1.0 / Math.sqrt(this.hiddenSize);
        for (Parameter weight : this.parameters())
            WeightInit.uniform(weight.data, -stdv, stdv);
    }

    @Override
    public String toString() {
        return "RNN(inputSize=" + inputSize + ", hiddenSize=" + hiddenSize + ")";
    }


    private class RNNImpl extends OperatorTensor {

        RNNImpl(Tensor input, Tensor h0) {
            INDArray hidden = Nd4j.create(input.data.shape()[0], input.data.shape()[1], hiddenSize);
            INDArray input_grad = Nd4j.emptyLike(input.data);
            INDArray weight_hh_grad = Nd4j.zerosLike(weight_hh.data);
            INDArray weight_ih_grad = Nd4j.zerosLike(weight_ih.data);
            INDArray bias_hh_grad = Nd4j.zerosLike(bias_hh.data);
            INDArray bias_ih_grad = Nd4j.zerosLike(bias_ih.data);

            OperandInfo[] operandInfos = {
                    new OperandInfo(input, () -> {
                        doBkwd(input.data, h0.data, hidden, this.grad, input_grad, weight_hh_grad, weight_ih_grad,
                                bias_hh_grad, bias_ih_grad);
                        return input_grad;
                    }),
                    new OperandInfo(weight_hh, () -> weight_hh_grad),
                    new OperandInfo(weight_ih, () -> weight_ih_grad),
                    new OperandInfo(bias_hh, () -> bias_hh_grad),
                    new OperandInfo(bias_ih, () -> bias_ih_grad),
            };

            Supplier<INDArray> forward = () -> doFwd(input.data, h0.data, hidden);

            OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

            setOperatorInfo(operatorInfo);
        }


        private INDArray doFwd(INDArray input, INDArray h0, INDArray hidden) {
            INDArray prevHidden = h0;
            long times = input.size(0);

            for (long i = 0; i < times; i++) {
                INDArray currIn = input.get(point(i), all(), all());
                INDArray currOut = hidden.get(point(i), all(), all());
                currOut.assign(currIn.mmul(weight_ih.data.transpose())
                        .add(prevHidden.mmul(weight_hh.data.transpose()))
                        .addRowVector(bias_hh.data)
                        .addRowVector(bias_ih.data)
                );
                IActivation activation = new ActivationTanH();
                activation.getActivation(currOut, true);
                prevHidden = currOut;
            }

            return hidden;
        }


        private void doBkwd(INDArray input, INDArray h0, INDArray hidden, INDArray epsilon, INDArray input_grad,
                            INDArray weight_hh_grad, INDArray weight_ih_grad, INDArray bias_hh_grad,
                            INDArray bias_ih_grad) {
            INDArray dzNext = null;
            for (long i = input.size(0) - 1; i >= 0; i--) {
                INDArray epsCurrent = epsilon.get(point(i), all(), all());
                INDArray hiddenCurrent = hidden.get(point(i), all(), all());
                INDArray inCurrent = input.get(point(i), all(), all());
                INDArray hiddenPrevious = (i == 0) ? h0 : hidden.get(point(i - 1), all(), all());
                INDArray epsOutCurrent = input_grad.get(point(i), all(), all());

                if (dzNext != null)
                    epsCurrent.addi(dzNext.mmul(weight_hh.data));

                INDArray dzCurrent = epsCurrent.mul(hiddenCurrent.mul(hiddenCurrent).mul(-1).add(1));

                epsOutCurrent.assign(dzCurrent.mmul(weight_ih.data));

                weight_hh_grad.addi(dzCurrent.transpose().mmul(hiddenPrevious));
                weight_ih_grad.addi(dzCurrent.transpose().mmul(inCurrent));
                bias_hh_grad.addi(dzCurrent.sum(0));
                bias_ih_grad.addi(dzCurrent.sum(0));

                dzNext = dzCurrent;
            }
        }

    }
}