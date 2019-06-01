package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.Supplier;

public class Reshape extends OperatorTensor {

    long[] from_shape;

    public Reshape(Tensor tensor, long... to_shape) {
        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> grad.reshape(from_shape))
        };

        Supplier<INDArray> forward = () -> {
            from_shape = tensor.data.shape();
            if (to_shape.length == 1 && to_shape[0] == -1) {
                return Nd4j.toFlattened(tensor.data);
            }
            return tensor.data.reshape(to_shape);
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
