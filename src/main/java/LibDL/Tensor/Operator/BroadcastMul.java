package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.function.Supplier;

/**
 * Only for <code>Conv2d</code>
 * @see LibDL.nn.Conv2d
 * */
public class BroadcastMul extends OperatorTensor {

    private static INDArray temp = Nd4j.zeros(1);

    /**
     * @param input tensor of shape(N, size*in_channel*out_channel, L)
     * @param weight tensor of shape(1, size*in_channel*out_channel, 1)
     * @param groups groups of <code>Conv2d</code>
     * */
    public BroadcastMul(Tensor input, Tensor weight, int in_channels, int out_channels, int groups) {

        assert input.data.rank() == 3;
        assert weight.data.rank() == 3;
        OperandInfo[] operandInfos = new OperandInfo[] {
                new OperandInfo(input, () -> grad.mul(weight.data.broadcast(input.data.shape()))),
                new OperandInfo(weight, () -> {
                    long size = weight.data.shape()[1] / in_channels / out_channels;
                    INDArray zeros = Nd4j.zerosLike(grad);
                    int step = in_channels / groups;
                    for (int i = 0; i < out_channels; i++) {
                        int bias = i / (out_channels / groups); // 1 / 2 = 0
                        long begin = i * in_channels + bias * step;
                        INDArrayIndex[] indArrayIndices = new INDArrayIndex[] {
                                NDArrayIndex.all(),
                                NDArrayIndex.interval(begin * size, (begin + step) * size),
                                NDArrayIndex.all()};
                        zeros.put(indArrayIndices, grad.get(indArrayIndices));
                    }
                    return zeros.mul(input.data).sum(0, 2).reshape(weight.data.shape());
                })
        };

      Supplier<INDArray> forward = () -> {

            long[] shape_i = input.data.shape();
            long[] shape_t = temp.shape();

            if (shape_t.length != shape_i.length) {
                temp = Nd4j.zerosLike(input.data);
            }else {
                if (shape_t[1] < shape_i[1]) {
                    temp = Nd4j.zeros(shape_t[0], shape_i[1], shape_t[2]);
                }
                if (shape_t[2] < shape_i[2]) {
                    temp = Nd4j.zeros(shape_t[0], shape_t[1], shape_i[2]);
                }
            }
            INDArray result = temp.get(
                    NDArrayIndex.interval(0, 1, shape_i[0]),
                    NDArrayIndex.interval(0, 1, shape_i[1]),
                    NDArrayIndex.interval(0, 1, shape_i[2]));

            Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(input.data, weight.data, result, 1));
            return result;
        };
//        Supplier<INDArray> forward = () -> {
//            Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(input.data, weight.data, input.data, 1));
//            return input.data;
//        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);
        setOperatorInfo(operatorInfo);
    }
}
