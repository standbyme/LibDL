package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.function.Supplier;

public class Unfold extends OperatorTensor {
    public Unfold(Tensor tensor, int kernel_size) {
        this(tensor, kernel_size, 0);
    }

    public Unfold(Tensor tensor, int kernel_size, int padding) {
        this(tensor, kernel_size, 0, 1);
    }

    public Unfold(Tensor tensor, int kernel_size, int padding, int stride) {

        // todo: only support kernel_size is a int, but PyTorch support Tuple


        assert padding==0;
        assert stride==1;

        int dilation = 1; // todo

        int filter_h = kernel_size;
        int filter_w = kernel_size;

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> {
                    assert false;
                    return null;
                }),
        };

        Supplier<INDArray> forward = () -> {

            assert tensor.out.rank() == 2;

            // todo: padding

            long input_h = tensor.out.shape()[0];
            long input_w = tensor.out.shape()[1];

            // todo: stride

            long amount_h = input_h - filter_h + 1;
            long amount_w = input_w - filter_w + 1;

            ArrayList<INDArray> buffer = new ArrayList<>((int) (amount_h * amount_w));

            for (int x = 0; x < amount_h; x++) {
                for (int y = 0; y < amount_w; y++) {
                    INDArray res = Nd4j.toFlattened(tensor.out.get(NDArrayIndex.interval(x, x + kernel_size), NDArrayIndex.interval(y, y + kernel_size)));
                    buffer.add(res);
                }
            }

            return Nd4j.vstack(buffer);

        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);
        setOperatorInfo(operatorInfo);
    }
}