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
     * Only for <code>Conv2d</code>
     * @see LibDL.nn.Conv2d
     * @param input tensor of shape(N, out, ah, aw)
     * @param B tensor of shape(out)
     * @param forConv2d for override */
    public BroadcastAdd(Tensor input, Tensor B, boolean forConv2d) {

        assert B.out.rank() == 1;

        OperandInfo[] operandInfos = {
                new OperandInfo(input, () -> dout),
                new OperandInfo(B, () -> dout.sum(0, 2, 3).reshape(dout.shape()[1]))
        };

        Supplier<INDArray> forward = () -> input.out.add(B.out
                .reshape(1, B.out.shape()[0], 1, 1)
                .broadcast(input.out.shape()));

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}