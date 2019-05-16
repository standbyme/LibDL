package LibDL.nn;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.Operator.Concat;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.Supplier;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;


public class BiRNN extends Module {

    // Layer configurations
    private long inputSize;
    private long hiddenSize;
    private long numLayer;

    // core
    private RNNBase fwd;
    private RNNBase bwd;

    public BiRNN(int inputSize, int hiddenSize) {
        this(inputSize, hiddenSize, 1);
    }

    public BiRNN(int inputSize, int hiddenSize, int numLayer) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.numLayer = numLayer;

        fwd = new RNN(inputSize, hiddenSize, numLayer);
        bwd = new RNN(inputSize, hiddenSize, numLayer);
    }

    private Tensor forward(Tensor input, Tensor h0) {
        // Reverse input, for the backward RNN.
        Tensor reverse = new Reverse(input);
        Tensor h0Fwd = h0.get(0);
        Tensor h0Bwd = h0.get(1);

        Tensor outFwd = fwd.forward(input, h0Fwd);

        Tensor outBwd = bwd.forward(reverse, h0Bwd);

        // Reverse the output of backward RNN, back to normal time series.
        outBwd = new Reverse(outBwd);
        return new Concat(2, outFwd, outBwd);
    }

    @Override
    public Tensor forward(Tensor input) {
        return null;
    }

    @Override
    public String toString() {
        return "BiRNN(inputSize=" + inputSize + ", hiddenSize=" + hiddenSize + ")";
    }

    private class Reverse extends OperatorTensor {
        Reverse (Tensor input) {
            OperandInfo[] operandInfos = {
                    new OperandInfo(input, () -> reverse(grad)),
            };

            Supplier<INDArray> forward = () -> reverse(input.data);

            OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

            setOperatorInfo(operatorInfo);
        }

        private INDArray reverse(INDArray input) {
            INDArray rev = Nd4j.zerosLike(input);
            long times = input.size(0);
            for (long i = 0; i < times; i++) {
                rev.get(point(i), all(), all()).assign(input.get(point(times - 1 - i), all(), all()));
            }
            return rev;
        }
    }
}