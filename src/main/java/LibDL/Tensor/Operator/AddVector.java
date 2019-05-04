package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class AddVector extends OperatorTensor {

    public AddVector(Tensor mat1, Tensor mat2) {

        OperandInfo[] operandInfos = {
                new OperandInfo(mat1, () -> grad),
                new OperandInfo(mat2, () -> grad.sum(0)),
        };

        Supplier<INDArray> forward = () -> mat1.data.addRowVector(mat2.data);

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