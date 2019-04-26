package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class Transpose extends OperatorTensor {

    public Transpose(Tensor tensor) {
        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> grad.transpose())
        };

        Supplier<INDArray> forward = () -> tensor.data.transpose();

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

}
