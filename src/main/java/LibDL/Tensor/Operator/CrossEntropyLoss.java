package LibDL.Tensor.Operator;

import LibDL.ND4JUtil;
import LibDL.Tensor.*;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class CrossEntropyLoss extends OperatorTensor {

    private final static double delta = Math.pow(10,-7);

    public CrossEntropyLoss(Tensor tensor, Tensor target) {
        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, null),
                new OperandInfo(target, null),
        };

        Supplier<INDArray> forward = () -> {
            INDArray y = tensor.out;
            INDArray t = target.out;

            return ND4JUtil.Log(y.add(delta)).muli(t).sum().muli(-1);
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
