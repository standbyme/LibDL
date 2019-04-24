package LibDL.Tensor.Operator;

import LibDL.ND4JUtil;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.function.Supplier;

public class Tanh extends OperatorTensor {
    public Tanh(Tensor tensor) {
        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> ND4JUtil.TanhDerivative(tensor.data).muli(grad)),
        };

        Supplier<INDArray> forward = () -> Transforms.tanh(tensor.data);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
