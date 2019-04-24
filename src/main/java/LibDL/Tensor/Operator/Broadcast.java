package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.function.Supplier;

public class Broadcast extends OperatorTensor {

    public Broadcast(Tensor mat, long... dim) {

        OperandInfo[] operandInfos = {
                new OperandInfo(mat, () -> grad.sum(Arrays.stream(dim)
                        .mapToInt(i -> (int) i).toArray())),
        };

        Supplier<INDArray> forward = () -> mat.data.broadcast(dim);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}