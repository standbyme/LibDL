package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.function.Supplier;

public class Unfold extends OperatorTensor {

    private long input_h;
    private long input_w;
    private long amount_h;
    private long amount_w;
    private long output_h;
    private long output_w;

    private Tensor input;

    private int[] stride;
    private int[] padding;
    private int[] dilation;

    private Unfold(Builder builder) {

        input = builder.input;
        int[] kernel_size = builder.kernel_size;
        stride = builder.stride;
        padding = builder.padding;
        dilation = builder.dilation;

        int filter_h = kernel_size[0];
        int filter_w = kernel_size[1];

        OperandInfo[] operandInfos = {

//                new OperandInfo(input, () -> {
//                    assert input.out.rank() == 4;
//                    INDArray out = input.out;
//                    long[] shape = out.shape();
//
//                    INDArray zeros = Nd4j.zeros(1, shape[2]+this.padding[0]*2, shape[3]+this.padding[1]*2);
//                    INDArray counts = Nd4j.zeros(1, shape[2]+this.padding[0]*2, shape[3]+this.padding[1]*2);
//
//                    for (long i = 0; i < amount_h; i++) {
//                        for (long j = 0; j < amount_w; j++) {
//                            zeros.put(new INDArrayIndex[] {
//                                    NDArrayIndex.all(),
//                                    NDArrayIndex.interval(i*stride[0], dilation[0], i*stride[0]+filter_h*dilation[0]),
//                                    NDArrayIndex.interval(j*stride[1], dilation[1], j*stride[1]+filter_w*dilation[1])
//                            }, 1);
//                            counts.addi(zeros);
//                            zeros.muli(0);
//                        }
//                    }
//                    counts = counts.get(
//                            NDArrayIndex.point(0),
//                            NDArrayIndex.interval(padding[0], padding[0]+shape[2]),
//                            NDArrayIndex.interval(padding[1], padding[1]+shape[3]));
//
//                    INDArray fold = Nd4j.zeros(shape[0], shape[1], shape[2]+this.padding[0]*2, shape[3]+this.padding[1]*2);
//                    INDArray column;
//                    for (long i = 0; i < amount_h; i++) {
//                        for (long j = 0; j < amount_w; j++) {
//                            column = dout.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i*amount_w+j))
//                                    .reshape(shape[0], shape[1], filter_h, filter_w);
//                            fold.put(new INDArrayIndex[] {
//                                    NDArrayIndex.all(), NDArrayIndex.all(),
//                                    NDArrayIndex.interval(i*stride[0], dilation[0], i*stride[0]+filter_h*dilation[0]),
//                                    NDArrayIndex.interval(j*stride[1], dilation[1], j*stride[1]+filter_w*dilation[1])
//                            }, column);
//                        }
//                    }
//                    fold = fold.get(
//                            NDArrayIndex.all(),
//                            NDArrayIndex.all(),
//                            NDArrayIndex.interval(padding[0], padding[0]+shape[2]),
//                            NDArrayIndex.interval(padding[1], padding[1]+shape[3]));
//
//                    return counts.reshape(1, 1, shape[2], shape[3]).broadcast(fold.shape()).muli(fold);
//                }),
                new OperandInfo(input, () -> {

                    assert input.out.rank() == 4;
                    INDArray out = input.out;
                    long[] shape = out.shape();

                    INDArray zeros = Nd4j.zeros(shape[0], shape[1], shape[2]+this.padding[0]*2, shape[3]+this.padding[1]*2);
                    INDArray result = zeros.dup();

                    INDArray column;
                    for (long i = 0; i < amount_h; i++) {
                        for (long j = 0; j < amount_w; j++) {
                            column = dout.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i*amount_w+j))
                                    .reshape(shape[0], shape[1], filter_h, filter_w);
                            zeros.put(new INDArrayIndex[] {
                                    NDArrayIndex.all(), NDArrayIndex.all(),
                                    NDArrayIndex.interval(i*stride[0], dilation[0], i*stride[0]+filter_h*dilation[0]),
                                    NDArrayIndex.interval(j*stride[1], dilation[1], j*stride[1]+filter_w*dilation[1])
                            }, column);
                            result.addi(zeros);
                            zeros.muli(0);
                        }
                    }

                    result = result.get(
                            NDArrayIndex.all(),
                            NDArrayIndex.all(),
                            NDArrayIndex.interval(padding[0], padding[0]+shape[2]),
                            NDArrayIndex.interval(padding[1], padding[1]+shape[3]));

                    return result;
                })
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

            for (long i = 0; i < amount_h; i++) {
                for (long j = 0; j < amount_w; j++) {
                    INDArray column = Nd4j.toFlattened(out.get(
                            NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.interval(i*stride[0], dilation[0], i*stride[0]+filter_h*dilation[0]),
                            NDArrayIndex.interval(j*stride[1], dilation[1], j*stride[1]+filter_w*dilation[1])));
                    result.put(new INDArrayIndex[] {
                                    NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i*amount_w+j)},
                            column.reshape(shape[0], 1, output_h));
                }
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
            if(kernel_size.length == 1)
                this.kernel_size = new int[] {kernel_size[0], kernel_size[0]};
            else
                this.kernel_size = kernel_size;
        }
        public Builder stride(int... stride) {
            if(stride.length == 1)
                this.stride = new int[] {stride[0], stride[0]};
            else
                this.stride = stride;
            return this;
        }
        public Builder padding(int... padding) {
            if(padding.length == 1)
                this.padding = new int[] {padding[0], padding[0]};
            else
                this.padding = padding;
            return this;
        }
        public Builder dilation(int... dilation) {
            if(dilation.length == 1)
                this.dilation = new int[] {dilation[0], dilation[0]};
            else
                this.dilation = dilation;
            return this;
        }
        public Unfold build() {
            return new Unfold(this);
        }
    }
}