package LibDL.Tensor.Operator;

import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class BroadcastAdd extends OperatorTensor {

    public BroadcastAdd(Tensor mat1, Tensor mat2) {

        OperandInfo[] operandInfos = {
                new OperandInfo(mat1,()-> grad),
                new OperandInfo(mat2,()-> grad.mean(0)),
        };

        Supplier<INDArray> forward = () -> mat1.data.addRowVector(mat2.data);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}