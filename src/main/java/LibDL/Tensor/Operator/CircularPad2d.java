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

public class CircularPad2d extends OperatorTensor {

    public CircularPad2d(Tensor input, int... padding) { // padding_left, padding_right, padding_top, padding_bottom

        assert input.data.rank() == 4 : "input.rank is expected to be 4";
        assert padding.length == 4 : "input.rank is expected to be 4";
        assert padding[0] <= input.data.shape()[2];
        assert padding[1] <= input.data.shape()[2];
        assert padding[2] <= input.data.shape()[3];
        assert padding[3] <= input.data.shape()[3];

        long[] shape = input.data.shape();

        OperandInfo[] operandInfos = {
                new OperandInfo(input, () -> {
                    INDArray result = grad.dup();
                    if (padding[3] > 0)
                        result.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(padding[2], padding[2] + padding[3]),
                                NDArrayIndex.all())
                                .addi(result.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                        NDArrayIndex.interval(padding[2] + shape[2], padding[2] + shape[2] + padding[3]),
                                        NDArrayIndex.all()));
                    if (padding[2] > 0)
                        result.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(padding[2] + shape[2] - padding[2], padding[2] + shape[2]),
                                NDArrayIndex.all())
                                .addi(result.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                        NDArrayIndex.interval(0, padding[2]),
                                        NDArrayIndex.all()));
                    if (padding[1] > 0)
                        result.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(padding[2], padding[2] + shape[2]),
                                NDArrayIndex.interval(padding[0], padding[0] + padding[1]))
                                .addi(result.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                        NDArrayIndex.interval(padding[2], padding[2] + shape[2]),
                                        NDArrayIndex.interval(padding[0] + shape[3], padding[0] + shape[3] + padding[1])));
                    if (padding[0] > 0)
                        result.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(padding[2], padding[2] + shape[2]),
                                NDArrayIndex.interval(padding[0] + shape[3] - padding[0], padding[0] + shape[3]))
                                .addi(result.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                        NDArrayIndex.interval(padding[2], padding[2] + shape[2]),
                                        NDArrayIndex.interval(0, padding[0])));

                    return result.get(NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.interval(padding[2], padding[2] + shape[2]),
                            NDArrayIndex.interval(padding[0], padding[0] + shape[3]));
                }),
        };

        Supplier<INDArray> forward = () -> {
            INDArray result = Nd4j.zeros(shape[0], shape[1],
                    shape[2] + padding[2] + padding[3], shape[3] + padding[0] + padding[1]);
            result.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.interval(padding[2], padding[2] + shape[2]),
                    NDArrayIndex.interval(padding[0], padding[0] + shape[3])}, input.data);
            if (padding[0] > 0)
                result.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(padding[2], padding[2] + shape[2]),
                                NDArrayIndex.interval(0, padding[0])},
                        input.data.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.all(),
                                NDArrayIndex.interval(shape[3] - padding[0], shape[3])));
            if (padding[1] > 0)
                result.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(padding[2], padding[2] + shape[2]),
                                NDArrayIndex.interval(padding[0] + shape[3], padding[0] + shape[3] + padding[1])},
                        input.data.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.all(),
                                NDArrayIndex.interval(0, padding[1])));
            if (padding[2] > 0)
                result.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(0, padding[2]),
                                NDArrayIndex.all()},
                        result.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(padding[2] + shape[2] - padding[2], padding[2] + shape[2]),
                                NDArrayIndex.all()));
            if (padding[3] > 0)
                result.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(padding[2] + shape[2], padding[2] + shape[2] + padding[3]),
                                NDArrayIndex.all()},
                        result.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                NDArrayIndex.interval(padding[2], padding[2] + padding[3]),
                                NDArrayIndex.all()));
            return result;
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);
        setOperatorInfo(operatorInfo);
    }
}
