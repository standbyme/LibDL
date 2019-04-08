package LibDL.Tensor.Operator;


import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class Average extends OperatorTensor {

    public Average(Tensor tensor) {

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> {
                    return dout.transpose().div((double) tensor.out.size(1)).repeat(1, tensor.out.size(1));
                }),
        };

        Supplier<INDArray> forward = () -> {
            // returns average value of every row
            return tensor.out.mean(1).transpose();
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}