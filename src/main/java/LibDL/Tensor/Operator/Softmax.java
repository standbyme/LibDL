package LibDL.Tensor.Operator;

import LibDL.ND4JUtil;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class Softmax extends OperatorTensor {
    public Softmax(Tensor tensor) {
        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, null),
        };

        Supplier<INDArray> forward = () -> {
            INDArray a = tensor.out;
            Number c = a.maxNumber();
            INDArray exp_a = ND4JUtil.Exp(a.sub(c));
            Number sum_exp_a = ND4JUtil.Exp(a.sub(c)).sumNumber();
            return exp_a.divi(sum_exp_a);

        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
