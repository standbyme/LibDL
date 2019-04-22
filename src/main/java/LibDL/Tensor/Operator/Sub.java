package LibDL.Tensor.Operator;

import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class Sub extends OperatorTensor {
    public Sub(Tensor mat1, Tensor mat2) {
        OperandInfo[] operandInfos = {
                new OperandInfo(mat1, () -> grad),
                new OperandInfo(mat2, () -> grad.mul(-1)),
        };

        Supplier<INDArray> forward = () -> mat1.data.sub(mat2.data);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
