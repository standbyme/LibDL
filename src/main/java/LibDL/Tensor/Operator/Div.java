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
                new OperandInfo(dividend, () -> dout.div(divisor.out)),
                new OperandInfo(divisor, () -> dividend.out.mul(dout).mul(-1.0)
                        .div(divisor.out.mul(divisor.out)))
        };

        Supplier<INDArray> forward = () -> dividend.out.div(divisor.out);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

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
