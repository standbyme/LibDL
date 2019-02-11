package LibDL.Tensor.Operator;

import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.Supplier;

public class Pow extends OperatorTensor {

    private static INDArray exec(TransformOp op) {
        if (op.x().isCleanedUp()) throw new IllegalStateException("NDArray already freed");
        return Nd4j.getExecutioner().execAndReturn(op);
    }

    private static INDArray pow(INDArray x, int exponent) {
        return exec(new org.nd4j.linalg.api.ops.impl.transforms.Pow(x, x.dup(), exponent));
    }

    public Pow(Tensor base, int exponent) {
        assert (exponent == 2);

        OperandInfo[] operandInfos = {
                new OperandInfo(base, () -> dout.mul(base.out.mul(2))),
        };

        Supplier<INDArray> forward = () -> pow(base.out, exponent);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}