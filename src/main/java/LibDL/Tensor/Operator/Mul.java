package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class Mul extends OperatorTensor {
    public Mul(Tensor tensor, Number times) {

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> grad.mul(times)),
        };

        Supplier<INDArray> forward = () -> tensor.data.mul(times);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

    public Mul(Tensor lhs, Tensor rhs) {
        OperandInfo[] operandInfos = {
                new OperandInfo(lhs, () -> grad.mul(rhs.data)),
                new OperandInfo(rhs, () -> grad.mul(lhs.data)),
        };

        Supplier<INDArray> forward = () -> lhs.data.mul(rhs.data);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}