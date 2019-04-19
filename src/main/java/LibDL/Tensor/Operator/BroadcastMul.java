package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;

import java.util.function.Supplier;

public class BroadcastMul extends OperatorTensor {

    /**
     * Only used in <code>Conv2d</code>
     * @param input tensor of shape(N, size*in_channel*out_channel, L)
     * @param weight tensor of shape(size*in_channel*out_channel)
     * */
    public BroadcastMul(Tensor input, Tensor weight) {

        OperandInfo[] operandInfos = new OperandInfo[] {
                new OperandInfo(input, () -> weight.out.broadcast(input.out)),
                new OperandInfo(weight, () -> input.out.sum(0, 2))
        };
        Supplier<INDArray> forward = () -> input.out.mul(weight.out.broadcast(input.out.dup()));

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);
        setOperatorInfo(operatorInfo);
    }
}
