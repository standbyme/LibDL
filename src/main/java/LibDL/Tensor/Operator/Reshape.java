package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class Reshape extends OperatorTensor {
    public Reshape(Tensor tensor, long... shape) {

        assert shape.length == 2;

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> {
                    assert false; // todo
                    return null;
                }),
        };

        Supplier<INDArray> forward = () -> {
            return tensor.out.reshape(shape);
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
