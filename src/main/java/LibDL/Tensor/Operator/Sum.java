package LibDL.Tensor.Operator;

import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.Supplier;


public class Sum extends OperatorTensor {

    public Sum(Tensor tensor) {

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> Nd4j.ones(tensor.out.shape())),
        };

        Supplier<INDArray> forward = () -> tensor.out.sum();

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}