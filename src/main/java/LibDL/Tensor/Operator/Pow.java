package LibDL.Tensor.Operator;

import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import LibDL.ND4JUtil;

import java.util.function.Supplier;

public class Pow extends OperatorTensor {

    public Pow(Tensor base, int exponent) {
        assert (exponent == 2);

        OperandInfo[] operandInfos = {
                new OperandInfo(base, () -> dout.mul(base.out.mul(2))),
        };

        Supplier<INDArray> forward = () -> ND4JUtil.pow(base.out, exponent);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}