package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.function.Supplier;

public class Reshape extends OperatorTensor {

    long[] from_shape;

    public Reshape(Tensor tensor, long... to_shape) {
        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> dout.reshape(from_shape))
        };

        Supplier<INDArray> forward = () -> {
            from_shape = tensor.out.shape();
            return tensor.out.reshape(to_shape);
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
    public Reshape(Tensor tensor, boolean isConv2d, int out_channel, int amount_h, int amount_w) {

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> dout.reshape(from_shape))
        };

        Supplier<INDArray> forward = () -> {
            long N = tensor.out.shape()[0];
            long sio = tensor.out.shape()[1];
            long L = tensor.out.shape()[2];

            long[] to_shape = new long[] {N, out_channel, sio/out_channel, amount_h, amount_w};
            from_shape = tensor.out.shape();
            return tensor.out.reshape(to_shape);
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
