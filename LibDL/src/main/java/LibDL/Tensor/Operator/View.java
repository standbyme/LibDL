package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class View extends OperatorTensor {
    public View(Tensor ori, int... shape) {
        OperandInfo[] operandInfos = {
                new OperandInfo(ori, () -> dout.reshape(ori.out.shape())),
        };

        Supplier<INDArray> forward = () -> ori.out.reshape(shape);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
