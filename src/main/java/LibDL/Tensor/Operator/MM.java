package LibDL.Tensor.Operator;

import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class MM extends OperatorTensor {

    public MM(Tensor mat1, Tensor mat2) {
        OperandInfo[] operandInfos = {
                new OperandInfo(mat1, () -> grad.mmul(mat2.data.transpose())),
                new OperandInfo(mat2, () -> mat1.data.transpose().mmul(grad)),
        };

        Supplier<INDArray> forward = () -> mat1.data.mmul(mat2.data);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}