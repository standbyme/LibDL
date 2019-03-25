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
                new OperandInfo(mat1,()->dout),
                new OperandInfo(mat2,()->dout.mean(0)),
        };

        Supplier<INDArray> forward = () -> mat1.out.addRowVector(mat2.out);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}