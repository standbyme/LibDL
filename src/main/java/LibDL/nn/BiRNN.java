package LibDL.nn;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.Operator.Concat;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.lang.Nullable;

import java.util.function.Supplier;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;


public class BiRNN extends Module {

    // Layer configurations
    public long inputSize;
    public long hiddenSize;
    public long numLayer;
    public RNNType rnnType;

    // core
    private RNNBase fwd;
    private RNNBase bwd;

    public BiRNN(int inputSize, int hiddenSize, int numLayer) {
        this(inputSize, hiddenSize, numLayer, RNNType.TYPE_RNN);
    }

    public BiRNN(int inputSize, int hiddenSize, RNNType rnnType) {
        this(inputSize, hiddenSize, 1, rnnType);
    }

    public BiRNN(int inputSize, int hiddenSize) {
        this(inputSize, hiddenSize, 1, RNNType.TYPE_RNN);
    }


    public BiRNN(int inputSize, int hiddenSize, int numLayer, RNNType rnnType) {
        this(inputSize, hiddenSize, numLayer, rnnType, 0, false, true);
    }

    public BiRNN(int inputSize, int hiddenSize, int numLayer, RNNType rnnType, double dropout, boolean relu, boolean bias) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.numLayer = numLayer;
        this.rnnType = rnnType;
        switch (rnnType) {
            case TYPE_GRU:
                fwd = new GRU(inputSize, hiddenSize, numLayer, relu, bias, dropout);
                bwd = new GRU(inputSize, hiddenSize, numLayer, relu, bias, dropout);
                break;
            case TYPE_RNN:
                fwd = new RNN(inputSize, hiddenSize, numLayer, relu, bias, dropout);
                bwd = new RNN(inputSize, hiddenSize, numLayer, relu, bias, dropout);
                break;
            case TYPE_LSTM:
                fwd = new LSTM(inputSize, hiddenSize, numLayer, relu, bias, dropout);
                bwd = new LSTM(inputSize, hiddenSize, numLayer, relu, bias, dropout);
                break;
        }
    }

    private Tensor forward(@NotNull Tensor input, @NotNull Tensor h0, @Nullable Tensor c0) {
        // Reverse input, for the backward RNN.
        Tensor reverse = new Reverse(input);
        Tensor h0Fwd = h0.get(0);
        Tensor h0Bwd = h0.get(1);


        Tensor outFwd = fwd.forward(input, h0Fwd, c0.get(0));

        Tensor outBwd = bwd.forward(reverse, h0Bwd, c0.get(1));

        // Reverse the output of backward RNN, back to normal time series.
        outBwd = new Reverse(outBwd);
        return new Concat(2, outFwd, outBwd);
    }

    public Tensor hn() {
        return new Concat(0, fwd.hn(), bwd.hn());
    }

    public Tensor cn() {
        if (rnnType != RNNType.TYPE_LSTM) {
            throw new UnsupportedOperationException("invalid call of cn()");
        }
        return new Concat(0, ((LSTM) fwd).cn(), ((LSTM) bwd).cn());
    }

    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException();
    }

    @Override
    public String toString() {
        return "Bi" + rnnType +
                "(inputSize=" + inputSize +
                ", hiddenSize=" + hiddenSize +
                ", numLayers=" + numLayer + ")";
    }

    private class Reverse extends OperatorTensor {
        Reverse(Tensor input) {
            OperandInfo[] operandInfos = {
                    new OperandInfo(input, () -> reverse(grad)),
            };

            Supplier<INDArray> forward = () -> reverse(input.data);

            OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

            setOperatorInfo(operatorInfo);
        }

        private INDArray reverse(INDArray input) {
            INDArray rev = Nd4j.emptyLike(input);
            long times = input.size(0);
            for (long i = 0; i < times; i++) {
                rev.get(point(i), all(), all()).assign(input.get(point(times - 1 - i), all(), all()));
            }
            return rev;
        }
    }
}