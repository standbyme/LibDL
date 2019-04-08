package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class Div extends OperatorTensor {

    public Div(Tensor dividend, int divisor) {
        assert (divisor != 0);

        OperandInfo[] operandInfos = {
                new OperandInfo(dividend, () -> dout.div(divisor)),
        };

        Supplier<INDArray> forward = () -> dividend.out.div(divisor);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

}
