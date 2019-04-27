package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.function.Supplier;

public class Pow extends OperatorTensor {

    public Pow(Tensor base, int exponent) {
//        assert (exponent > 0);

        OperandInfo[] operandInfos = {
                new OperandInfo(base, () -> grad.mul(Transforms.pow(base.data, exponent - 1).mul(exponent))),
        };

        Supplier<INDArray> forward = () -> Transforms.pow(base.data, exponent);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}