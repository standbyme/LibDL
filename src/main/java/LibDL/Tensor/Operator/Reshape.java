package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class Reshape extends OperatorTensor {

    long[] from_shape;

    public Reshape(Tensor tensor, long... to_shape) {
        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> dout.reshape(from_shape))
        };

        Supplier<INDArray> forward = () -> {
            from_shape = tensor.out.shape();
            return tensor.out.reshape(to_shape);
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
