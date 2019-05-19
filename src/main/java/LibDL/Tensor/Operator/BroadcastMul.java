package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.bytedeco.javacpp.FloatPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
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
                new OperandInfo(input, () -> {
                    INDArray temp = expandAndReturnTemp(grad.shape());
                    Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(grad, weight.data, temp, 1));
                    return temp;
                }),
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
                }),
//                new OperandInfo(weight, () -> {
//                    long size = weight.data.shape()[1] / in_channels / out_channels;
//                    INDArray zeros = Nd4j.zerosLike(grad);
//                    int step = in_channels / groups;
//                    float[] f = new float[(int) (size * step * grad.shape()[0] * grad.shape()[2])];
//                    FloatPointer floatPointerX, floatPointerY;
//
//                    INDArrayIndex[] indArrayIndices = new INDArrayIndex[] {
//                            NDArrayIndex.all(),
//                            null,
//                            NDArrayIndex.all()};
//                    for (int i = 0; i < out_channels; i++) {
//                        int bias = i / (out_channels / groups); // 1 / 2 = 0
//                        long begin = i * in_channels + bias * step;
//                        indArrayIndices[1] = NDArrayIndex.interval(begin * size, (begin + step) * size);
//
//                        floatPointerX = (FloatPointer) grad.get(indArrayIndices).data().pointer();
//                        floatPointerY = (FloatPointer) zeros.get(indArrayIndices).data().pointer();
//                        floatPointerX.get(f, 0, f.length);
//                        floatPointerY.put(f, 0, f.length);
//                    }
//                    return zeros.mul(input.data).sum(0, 2).reshape(weight.data.shape());
//                })
        };

        Supplier<INDArray> forward = () -> {
            INDArray result = expandAndReturnTemp(input.data.shape());
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

    private INDArray expandAndReturnTemp(long[] shape_i) {
        long[] shape_t = temp.shape();

        if (shape_t.length != shape_i.length) {
            System.out.print(Arrays.toString(temp.shape()) + " >>> ");
            temp = Nd4j.zeros(shape_i); // TODO
            System.out.println("new:" + "temp:shape:" + Arrays.toString(temp.shape()));
            shape_t = temp.shape();
        }else {
            int n = shape_t.length;
            long[] expension = Arrays.copyOf(shape_t, n);
            for (int i = 0; i < n; i++) {
                if (shape_t[i] < shape_i[i]) {
                    expension[i] = shape_i[i];
                    System.out.print(Arrays.toString(temp.shape()) + " >>> ");
                    temp = Nd4j.zeros(expension);
                    System.out.println("new:" + i + "temp:shape:" + Arrays.toString(temp.shape()));
                    shape_t = temp.shape();
                }
            }
        }
        INDArrayIndex[] indArrayIndices = new INDArrayIndex[shape_t.length];
        for (int i = 0; i < shape_t.length; i++) {
            indArrayIndices[i] = NDArrayIndex.interval(0, 1, shape_i[i]);
        }
        return temp.get(indArrayIndices);
    }
}
