package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class Sub extends OperatorTensor {
    public Sub(Tensor lhs, Tensor rhs) {
        OperandInfo[] operandInfos = {
                new OperandInfo(lhs, () -> grad),
                new OperandInfo(rhs, () -> grad.mul(-1)),
        };

        Supplier<INDArray> forward = () -> lhs.data.sub(rhs.data);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

    public Sub(Tensor lhs, Number rhs) {
        OperandInfo[] operandInfos = {
                new OperandInfo(lhs, () -> grad),
        };

        Supplier<INDArray> forward = () -> lhs.data.sub(rhs);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

}
