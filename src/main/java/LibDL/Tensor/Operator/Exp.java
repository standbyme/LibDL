package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.function.Supplier;

public class Exp extends OperatorTensor {
    public Exp(Tensor tensor) {
        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> this.data.muli(grad))
        };

        Supplier<INDArray> forward = () -> Transforms.exp(tensor.data);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
