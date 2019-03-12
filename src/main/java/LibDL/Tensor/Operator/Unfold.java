package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;

import java.util.function.Supplier;

public class Unfold extends OperatorTensor {
    public Unfold(Tensor tensor, int kernel_size) {
        this(tensor, kernel_size, 1);
    }

    public Unfold(Tensor tensor, int kernel_size, int padding) {

        // todo: only support kernel_size is a int, but PyTorch support Tuple

        int dilation = 1; // todo
        int stride = 1;
        int filter_h = kernel_size;
        int filter_w = kernel_size;

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> {
                    assert false;
                    return null;
                }),
        };

        Supplier<INDArray> forward = () -> Convolution.im2col(tensor.out, filter_h, filter_w, stride, stride, padding, padding, 0, false);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);
        setOperatorInfo(operatorInfo);
    }
}