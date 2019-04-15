package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
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

    private Tensor input;

    private int[] kernel_size;
    private int[] stride = {1, 1};
    private int[] padding = {0, 0};
    private int[] dilation = {1, 1};

    public Unfold(Builder builder) {

        input = builder.input;

        kernel_size = builder.kernel_size;
        stride = builder.stride;
        padding = builder.padding;
        dilation = builder.dilation;

        int filter_h = kernel_size[0];
        int filter_w = kernel_size[1];


        OperandInfo[] operandInfos = {
                new OperandInfo(input, () -> {
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

            assert input.out.rank() == 4;
            INDArray out = input.out;
            long[] shape = out.shape();

            if(padding[0] + padding[1] > 0) {
                INDArray padding = Nd4j.zeros(
                        shape[0], shape[1], shape[2]+this.padding[0]*2, shape[3]+this.padding[1]*2);
                padding.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(this.padding[0], this.padding[0]+shape[2]),
                        NDArrayIndex.interval(this.padding[1], this.padding[1]+shape[3])).assign(out);
                out = padding;
            }

            shape = out.shape();
            input_h = shape[2];
            input_w = shape[3];
            long channel = shape[1];

            int _filter_h = (filter_h - 1) * dilation[0] + 1;
            int _filter_w = (filter_w - 1) * dilation[1] + 1;
            amount_h = (input_h - _filter_h) / stride[0] + 1;
            amount_w = (input_w - _filter_w) / stride[1] + 1;

            output_h = channel * filter_h * filter_w;
            output_w = amount_h * amount_w;

            INDArray result = Nd4j.zeros(shape[0], output_h, output_w);
            for(int batch = 0; batch < shape[0]; batch++) {
                ArrayList<INDArray> buffer = new ArrayList<>((int) output_w);
                for(long c = 0; c < channel; c++) {
                    INDArray m = Nd4j.zeros(output_w, filter_h * filter_w);
                    for (long i = 0; i < amount_h; i++) {
                        for (long j = 0; j < amount_w; j++) {
                            INDArray column = Nd4j.toFlattened(out.get(
                                    NDArrayIndex.indices(batch), NDArrayIndex.indices(c),
                                    NDArrayIndex.interval(i*stride[0], dilation[0], i*stride[0]+filter_h*dilation[0]),
                                    NDArrayIndex.interval(j*stride[1], dilation[1], j*stride[1]+filter_w*dilation[1])));
                            m.putRow(i*amount_w+j, column);
                        }
                    }
                    buffer.add(m.transpose());
                }
                result.put(new INDArrayIndex[] {NDArrayIndex.point(batch), NDArrayIndex.all(), NDArrayIndex.all()},
                        Nd4j.vstack(buffer));
            }
            return result;
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);
        setOperatorInfo(operatorInfo);
    }

    public static class Builder {
        private Tensor input;
        private int[] kernel_size;
        private int[] stride = {1, 1};
        private int[] padding = {0, 0};
        private int[] dilation = {1, 1};
        public Builder(Tensor input, int... kernel_size) {
            this.input = input;
            this.kernel_size = kernel_size;
        }
        public Builder stride(int... stride) {
            this.stride = stride;
            return this;
        }
        public Builder padding(int... padding) {
            this.padding = padding;
            return this;
        }
        public Builder dilation(int... dilation) {
            this.dilation = dilation;
            return this;
        }
        public Unfold build() {
            return new Unfold(this);
        }
    }
}