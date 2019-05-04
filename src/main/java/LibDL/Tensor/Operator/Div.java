package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class Div extends OperatorTensor {

    public Div(Tensor dividend, Tensor divisor) {

        OperandInfo[] operandInfos = {
                new OperandInfo(dividend, () -> grad.div(divisor.data)),
                new OperandInfo(divisor, () -> dividend.data.mul(grad).mul(-1.0)
                        .div(divisor.data.mul(divisor.data)))
        };

        Supplier<INDArray> forward = () -> dividend.data.div(divisor.data);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

    public Div(Tensor dividend, Number divisor) {
        assert (!divisor.equals(0));

        OperandInfo[] operandInfos = {
                new OperandInfo(dividend, () -> grad.div(divisor)),
        };

        Supplier<INDArray> forward = () -> dividend.data.div(divisor);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

}
