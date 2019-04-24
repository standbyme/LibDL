package LibDL.Tensor.Operator;

import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class BroadcastAdd extends OperatorTensor {

    public BroadcastAdd(Tensor mat1, Tensor mat2) {

        OperandInfo[] operandInfos = {
                new OperandInfo(mat1,()->dout),
                new OperandInfo(mat2,()->dout.mean(0)),
        };

        Supplier<INDArray> forward = () -> mat1.out.addRowVector(mat2.out);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

    /**
     * Only for <code>Conv2d</code>*/
    public BroadcastAdd(Tensor input, Tensor bias, boolean isConv2d) {

        OperandInfo[] operandInfos = {
                new OperandInfo(input, () -> null)
        };

        Supplier<INDArray> forward = () -> input.out.add(bias.out
                .reshape(1, bias.out.shape()[0], 1, 1)
                .broadcast(input.out.shape()));

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}