package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class Self extends OperatorTensor {

    public Self(Tensor tensor) {
        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> grad),
        };

        Supplier<INDArray> forward = () -> tensor.data;

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
