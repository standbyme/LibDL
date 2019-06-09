package LibDL.Tensor.Operator;

import LibDL.Tensor.*;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class EmbeddingBackward extends OperatorTensor {
    public EmbeddingBackward(Tensor tensor, Tensor index,
                             long num_weights,
                             long padding_idx,
                             boolean scale_grad_by_freq//False
    ) {
        OperandInfo[] operandInfos = {
//                new OperandInfo(tensor, () -> {
//                    long numel = index.numel();
//                    Tensor grad = new Variable(this.grad.reshape(numel, this.grad.size(-1)));
//                    Tensor grad_weight = Tensor.zeros(num_weights, this.grad.size(this.grad.rank() - 1));
//
//                }
        };

        Supplier<INDArray> forward = () -> {
            return index.data;
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
