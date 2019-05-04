package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;
import java.util.stream.IntStream;

public class AddVector extends OperatorTensor {

    public AddVector(Tensor tensor, Tensor vector) {

        int rank = tensor.data.rank();
        int[] dims = IntStream.range(0, rank - 1).toArray();

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> grad),
                new OperandInfo(vector, () -> grad.sum(dims)),
        };

        long[] newShape = new long[rank];
        for(int i = 0; i < rank - 1; i++)
            newShape[i] = 1;
        newShape[rank - 1] = vector.data.length();

        Supplier<INDArray> forward = () -> tensor.data.add(vector.data
                .reshape(newShape)
                .broadcast(tensor.data.shape()));

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

    /**
     * Only for <code>Conv2d</code>
     *
     * @param input     tensor of shape(N, out, ah, aw)
     * @param B         tensor of shape(out)
     * @param forConv2d for override
     * @see LibDL.nn.Conv2d
     */
    public AddVector(Tensor input, Tensor B, boolean forConv2d) {

        assert B.data.rank() == 1;

        OperandInfo[] operandInfos = {
                new OperandInfo(input, () -> grad),
                new OperandInfo(B, () -> grad.sum(0, 2, 3).reshape(grad.shape()[1]))
        };

        Supplier<INDArray> forward = () -> input.data.add(B.data
                .reshape(1, B.data.shape()[0], 1, 1)
                .broadcast(input.data.shape()));

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}