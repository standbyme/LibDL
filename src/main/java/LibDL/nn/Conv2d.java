package LibDL.nn;

import LibDL.Tensor.Operator.Reshape;
import LibDL.Tensor.Operator.*;
import LibDL.Tensor.Parameter;
import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class Conv2d extends Module {

    private int in_channels;
    private int out_channels;
    private int[] kernel_size;
    private int[] stride;
    private int[] padding;
    private int[] dilation;
    private int groups;
    private boolean bias;

    // TODO `W` and `B` should be modified by `final`
    private Parameter W;
    private Parameter B;

    // TODO  can be removed. These fields are only for testing
    public INDArray data;
    public INDArray grad;
    public Tensor core;

    private Conv2d(Builder builder) {
        this.in_channels = builder.in_channels;
        this.out_channels = builder.out_channels;
        this.kernel_size = builder.kernel_size;
        this.stride = builder.stride;
        this.padding = builder.padding;
        dilation = builder.dilation;
        groups = builder.groups;
        bias = builder.bias;

        assert (in_channels % groups == 0 && out_channels % groups == 0);

        INDArray zeros = Nd4j.zeros(out_channels, in_channels, kernel_size[0], kernel_size[1]);
        for (int i = 0; i < groups; i++) {
            for (int j = 0; j < out_channels / groups; j++) {
                INDArray w = Nd4j.rand(new int[]{in_channels / groups, kernel_size[0], kernel_size[1]}).subi(0.5);
                zeros.put(new INDArrayIndex[]{NDArrayIndex.point(i * out_channels / groups + j),
                        NDArrayIndex.interval(i * in_channels / groups, (i + 1) * in_channels / groups),
                        NDArrayIndex.all(), NDArrayIndex.all()}, w);
            }
        }
        W = new Parameter(zeros);

        if (bias) {
            B = new Parameter(Nd4j.rand(new int[]{out_channels}).reshape(out_channels).subi(0.5));
        } else {
            B = null;
        }
    }

    // TODO can be removed. This function is only for testing, and may need more tests
    public void setW(INDArray value) {
        INDArray zeros = Nd4j.zeros(out_channels, in_channels, kernel_size[0], kernel_size[1]);
        for (int i = 0; i < groups; i++) {
            for (int j = 0; j < out_channels / groups; j++) {
                INDArray w = value.get(NDArrayIndex.point(i * out_channels / groups + j), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all());
                zeros.put(new INDArrayIndex[]{NDArrayIndex.point(i * out_channels / groups + j),
                        NDArrayIndex.interval(i * in_channels / groups, (i + 1) * in_channels / groups),
                        NDArrayIndex.all(), NDArrayIndex.all()}, w);
            }
        }
        W = new Parameter(zeros);
    }

    // TODO can be removed. This function is only for testing
    public void setB(INDArray value) {
        B = new Parameter(value);
    }

    // TODO can be removed. This function is only for testing
    public void backward() {
        core.grad = grad;
        core.backward();
    }

    // TODO can be removed. This function is only for testing
    public Tensor apply(Tensor input) {
        core = forward(input);
        data = core.data;
        return core;
    }

    public Variable getW() {
        return W;
    }

    public Variable getB() {
        return B;
    }

    @Override
    public Tensor forward(Tensor input) {
        int _filter_h = (kernel_size[0] - 1) * dilation[0] + 1;
        int _filter_w = (kernel_size[1] - 1) * dilation[1] + 1;
        long amount_h = (input.data.shape()[2] + padding[0] * 2 - _filter_h) / stride[0] + 1;
        long amount_w = (input.data.shape()[3] + padding[1] * 2 - _filter_w) / stride[1] + 1;
        Unfold unfold = new Unfold.Builder(input, kernel_size)
                .padding(padding)
                .stride(stride)
                .dilation(dilation)
                .build();
        BroadcastMul broadcastMul = new BroadcastMul(
                new Concat(unfold, out_channels, 1),
                new Reshape(W, 1, in_channels * out_channels * kernel_size[0] * kernel_size[1], 1),
                in_channels, out_channels, groups);
        Sum sum = new Sum(new Reshape(broadcastMul,
                broadcastMul.data.shape()[0], out_channels,
                kernel_size[0] * kernel_size[1] * in_channels, amount_h, amount_w), 2);
        if (bias) {
            return new AddVector(sum, B, true);
        } else {
            return sum;
        }
    }

    public static class Builder {
        private int in_channels;
        private int out_channels;
        private int[] kernel_size;
        private int[] stride = {1, 1};
        private int[] padding = {0, 0};
        private int[] dilation = {1, 1};
        private int groups = 1;
        private boolean bias = true;

        public Builder(int in_channels, int out_channels, int... kernel_size) {
            this.in_channels = in_channels;
            this.out_channels = out_channels;
            if (kernel_size.length == 1) {
                this.kernel_size = new int[]{kernel_size[0], kernel_size[0]};
            } else {
                this.kernel_size = kernel_size;
            }
        }

        public Builder stride(int... stride) {
            if (stride.length == 1)
                this.stride = new int[]{stride[0], stride[0]};
            else
                this.stride = stride;
            return this;
        }

        public Builder padding(int... padding) {
            if (padding.length == 1)
                this.padding = new int[]{padding[0], padding[0]};
            else
                this.padding = padding;
            return this;
        }

        public Builder dilation(int... dilation) {
            if (dilation.length == 1)
                this.dilation = new int[]{dilation[0], dilation[0]};
            else
                this.dilation = dilation;
            return this;
        }

        public Builder groups(int groups) {
            this.groups = groups;
            return this;
        }

        public Builder bias(boolean bias) {
            this.bias = bias;
            return this;
        }

        public Conv2d build() {
            return new Conv2d(this);
        }
    }
}
