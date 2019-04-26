package LibDL.Tensor.Operator;

import LibDL.ND4JUtil;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class ReLU extends OperatorTensor {
    public ReLU(Tensor tensor) {
        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> ND4JUtil.Step(tensor.data).muli(grad)),
        };

        Supplier<INDArray> forward = () -> ND4JUtil.ReLU(tensor.data);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
