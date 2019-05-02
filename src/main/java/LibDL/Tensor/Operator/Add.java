package LibDL.Tensor.Operator;

import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class Add extends OperatorTensor {
    public Add(Tensor mat1, Tensor mat2) {
        OperandInfo[] operandInfos = {
                new OperandInfo(mat1, () -> grad),
                new OperandInfo(mat2, () -> grad),
        };

        Supplier<INDArray> forward = () -> mat1.data.add(mat2.data);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

    public Add(Tensor lhs, Number rhs) {
        this(lhs, Tensor.numbers(rhs, lhs.sizes()));
    }
}