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

    private long input_h;
    private long input_w;
    private long amount_h;
    private long amount_w;
    private long output_h;
    private long output_w;

    public Unfold(Tensor tensor, int kernel_size) {
        this(tensor, kernel_size, 0);
    }

    public Unfold(Tensor tensor, int kernel_size, int padding) {
        this(tensor, kernel_size, 0, 1);
    }

    public Unfold(Tensor tensor, int kernel_size, int padding, int stride) {

        // todo: only support kernel_size is a int, but PyTorch support Tuple


        assert padding == 0;
        assert stride == 1;

        int dilation = 1; // todo

        int filter_h = kernel_size;
        int filter_w = kernel_size;

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> {
                    double[][] result = new double[(int) input_h][(int) input_w];
                    for (int x = 0; x < output_h; x++) {
                        long base_x = x / amount_w;
                        long base_y = x % amount_w;
                        for (int y = 0; y < output_w; y++) {
                            long offset_x = y / filter_w;
                            long offset_y = y % filter_w;

                            result[(int) (base_x + offset_x)][(int) (base_y + offset_y)] += dout.getDouble(x, y);
                        }
                    }
                    return Nd4j.create(result);
                }),
        };

        Supplier<INDArray> forward = () -> {

            assert tensor.out.rank() == 2;

            // todo: padding

            input_h = tensor.out.shape()[0];
            input_w = tensor.out.shape()[1];

            // todo: stride

            amount_h = input_h - filter_h + 1;
            amount_w = input_w - filter_w + 1;

            output_h = amount_h * amount_w;
            output_w = filter_h * filter_w;

            ArrayList<INDArray> buffer = new ArrayList<>((int) (output_h));

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