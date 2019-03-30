package LibDL.Tensor.Operator;

import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.function.Supplier;



public class Log extends OperatorTensor {

    public Log(Tensor tensor) {
        OperandInfo[] operandInfos = {
            new OperandInfo(tensor, () -> Transforms.pow(dout, -1))
        };

        Supplier<INDArray> forward = () -> Transforms.log(tensor.out);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
