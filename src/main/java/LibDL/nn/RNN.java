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

    private Variable _h0;

    public RNN(int inputSize, int hiddenSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        weight_hh = new Parameter(Nd4j.create(hiddenSize, hiddenSize));
        weight_ih = new Parameter(Nd4j.create(inputSize, hiddenSize));
        bias_hh = new Parameter(Nd4j.create(1, hiddenSize));
        bias_ih = new Parameter(Nd4j.create(1, hiddenSize));

        resetParameters();
    }

    public Tensor forward(Tensor input) {
        return new RNNImpl(input, _h0);
    }

    public void setH0(Variable h0) {
        this._h0 = h0;
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

    private class RNNImpl extends OperatorTensor {
        private Tensor input;
        private Tensor h0;

        private Tensor hidden;
        private INDArray input_grad, weight_hh_grad, weight_ih_grad, bias_hh_grad, bias_ih_grad;

        private IActivation activation = new ActivationTanH();

        RNNImpl(Tensor input, Tensor h0) {
            this.input = input;
            this.h0 = h0;

            OperandInfo[] operandInfos = {
                    new OperandInfo(input, () -> {
                        backwardHelper(this.grad);
                        return input_grad;
                    }),
                    new OperandInfo(weight_hh, () -> weight_hh_grad),
                    new OperandInfo(weight_ih, () -> weight_ih_grad),
                    new OperandInfo(bias_hh, () -> bias_hh_grad),
                    new OperandInfo(bias_ih, () -> bias_ih_grad),
            };

            Supplier<INDArray> forward = this::forwardHelper;

            OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

            setOperatorInfo(operatorInfo);
        }

        INDArray forwardHelper() {
            hidden = new Variable(Nd4j.create(input.data.shape()[0], input.data.shape()[1], hiddenSize), true);
            INDArray prevHidden = h0.data;
            long times = input.data.size(0);

            for (long i = 0; i < times; i++) {
                INDArray currIn = input.data.get(point(i), all(), all());
                INDArray currOut = hidden.data.get(point(i), all(), all());
                currOut.assign(currIn.mmul(weight_ih.data.transpose())
                        .add(prevHidden.mmul(weight_hh.data.transpose()))
                        .addRowVector(bias_hh.data)
                        .addRowVector(bias_ih.data)
                );
                activation.getActivation(currOut, true);
                prevHidden = currOut;
            }

            return hidden.data;
        }

        void backwardHelper(INDArray epsilon) {
            input_grad = Nd4j.emptyLike(input.data);
            weight_hh_grad = Nd4j.zerosLike(weight_hh.data);
            weight_ih_grad = Nd4j.zerosLike(weight_ih.data);
            bias_hh_grad = Nd4j.zerosLike(bias_hh.data);
            bias_ih_grad = Nd4j.zerosLike(bias_ih.data);

            INDArray dzNext = null;
            for (long i = input.data.size(0) - 1; i >= 0; i--) {
                INDArray epsCurrent = epsilon.get(point(i), all(), all());
                INDArray hiddenCurrent = hidden.data.get(point(i), all(), all());
                INDArray inCurrent = input.data.get(point(i), all(), all());
                INDArray hiddenPrevious = (i == 0) ? h0.data : hidden.data.get(point(i - 1), all(), all());
                INDArray epsOutCurrent = input_grad.get(point(i), all(), all());

                if (dzNext != null)
                    epsCurrent.addi(dzNext.mmul(weight_hh.data));

                INDArray dzCurrent = epsCurrent.mul(hiddenCurrent.mul(hiddenCurrent).mul(-1).add(1));

                epsOutCurrent.assign(dzCurrent.mmul(weight_ih.data));

                weight_ih_grad.addi(dzCurrent.transpose().mmul(inCurrent));
                weight_hh_grad.addi(dzCurrent.transpose().mmul(hiddenPrevious));
                bias_hh_grad.addi(dzCurrent.sum(0));
                bias_ih_grad.addi(dzCurrent.sum(0));

                dzNext = dzCurrent;
            }

        }
    }
}