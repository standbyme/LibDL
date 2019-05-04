package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.function.Supplier;

public class BroadcastMul extends OperatorTensor {

    /**
     * Only for <code>Conv2d</code>
     *
     * @param input  tensor of shape(N, size*in_channel*out_channel, L)
     * @param weight tensor of shape(1, size*in_channel*out_channel, 1)
     * @param groups groups of <code>Conv2d</code>
     * @see LibDL.nn.Conv2d
     */
    public BroadcastMul(Tensor input, Tensor weight, int in_channels, int out_channels, int groups) {

        assert input.data.rank() == 3;
        assert weight.data.rank() == 3;

        OperandInfo[] operandInfos = new OperandInfo[]{
                new OperandInfo(input, () -> grad.mul(weight.data.broadcast(input.data.shape()))),
                new OperandInfo(weight, () -> {
                    long size = weight.data.shape()[1] / in_channels / out_channels;
                    INDArray zeros = Nd4j.zerosLike(grad);
                    int step = in_channels / groups;
                    for (int i = 0; i < out_channels; i++) {
                        int bias = i / (out_channels / groups); // 1 / 2 = 0
                        long begin = i * in_channels + bias * step;
                        INDArrayIndex[] indArrayIndices = new INDArrayIndex[]{
                                NDArrayIndex.all(),
                                NDArrayIndex.interval(begin * size, (begin + step) * size),
                                NDArrayIndex.all()};
                        zeros.put(indArrayIndices, grad.get(indArrayIndices));
                    }
                    return zeros.mul(input.data).sum(0, 2).reshape(weight.data.shape());
                })
        };
        Supplier<INDArray> forward = () -> input.data.mul(weight.data.broadcast(input.data.shape()));

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);
        setOperatorInfo(operatorInfo);
    }
}
