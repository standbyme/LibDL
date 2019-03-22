package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.Supplier;


public class Max extends OperatorTensor {

    public Max(Tensor tensor) {

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> {
                    assert false; // todo
                    return null;
                }),
        };

        Supplier<INDArray> forward = () -> tensor.out.max();

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}