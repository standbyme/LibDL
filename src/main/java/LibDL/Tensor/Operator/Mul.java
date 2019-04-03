package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class Mul extends OperatorTensor {
    public Mul(Tensor mat1, Tensor mat2) {
        OperandInfo[] operandInfos = {
                new OperandInfo(mat1, () -> mat2.out),
                new OperandInfo(mat2, () -> mat1.out),
        };

        Supplier<INDArray> forward = () -> mat1.out.mul(mat2.out);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}