package LibDL.nn;

import LibDL.Tensor.Operator.Reshape;
import LibDL.Tensor.Operator.*;
import LibDL.Tensor.Parameter;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class Conv2d_N extends Module {

    private int in_channels;
    private int out_channels;
    private int[] kernel_size;
    private int[] stride;
    private int[] padding;
    private String padding_mode;
    private int[] dilation;
    private int groups;
    private boolean bias;

    private Parameter W;
    private Parameter B;

    private Conv2d_N(Builder builder) {
        this.in_channels = builder.in_channels;
        this.out_channels = builder.out_channels;
        this.kernel_size = builder.kernel_size;
        this.stride = builder.stride;
        this.padding = builder.padding;
        padding_mode = builder.padding_mode;
        dilation = builder.dilation;
        groups = builder.groups;
        bias = builder.bias;

        init();
    }

    public Conv2d_N(int in_channels, int out_channels, int[] kernel_size,
                    int[] stride, int[] padding, String padding_mode, int[] dilation, int groups, boolean bias) {
        this.in_channels = in_channels;
        this.out_channels = out_channels;
        this.kernel_size = kernel_size;
        this.stride = stride;
        this.padding = padding;
        this.padding_mode = padding_mode;
        this.dilation = dilation;
        this.groups = groups;
        this.bias = bias;

        init();
    }

    private void init() {
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

    public Parameter getW() {
        return W;
    }

    public Parameter getB() {
        return B;
    }

    @Override
    public Tensor forward(Tensor input) {
        int _filter_h = (kernel_size[0] - 1) * dilation[0] + 1;
        int _filter_w = (kernel_size[1] - 1) * dilation[1] + 1;
        long amount_h;
        long amount_w;

        Unfold unfold;
        if(this.padding_mode.equals("circular")) {
            amount_h = (input.data.shape()[2] + (padding[0] + 1) / 2 + padding[0] / 2 - _filter_h) / stride[0] + 1;
            amount_w = (input.data.shape()[3] + (padding[1] + 1) / 2 + padding[1] / 2 - _filter_w) / stride[1] + 1;
            CircularPad2d circularPad2d = new CircularPad2d(input,
                    (padding[1] + 1) / 2, padding[1] / 2, (padding[0] + 1) / 2, padding[0] / 2); // TODO /
            unfold = new Unfold.Builder(circularPad2d, kernel_size)
                    .padding(0)
                    .stride(stride)
                    .dilation(dilation)
                    .build();
        } else {
            amount_h = (input.data.shape()[2] + padding[0] * 2 - _filter_h) / stride[0] + 1;
            amount_w = (input.data.shape()[3] + padding[1] * 2 - _filter_w) / stride[1] + 1;
            unfold = new Unfold.Builder(input, kernel_size)
                    .padding(padding)
                    .stride(stride)
                    .dilation(dilation)
                    .build();
        }

        Reshape reshape = new Reshape(W, 1, in_channels * out_channels * kernel_size[0] * kernel_size[1], 1);

        Correlation m = new Correlation(unfold, reshape, amount_h, amount_w, in_channels, out_channels, groups);
//        BroadcastMul broadcastMul = new BroadcastMul(
//                new Concat(unfold, out_channels, 1),
//                reshape,
//                in_channels, out_channels, groups);
//
//        Sum sum = new Sum(new Reshape(broadcastMul,
//                broadcastMul.data.shape()[0], out_channels,
//                kernel_size[0] * kernel_size[1] * in_channels, amount_h, amount_w), 2);

        if (bias) {
            return new AddVector(m, B, true);
        } else {
            return m;
        }
    }

    public static class Builder {
        private int in_channels;
        private int out_channels;
        private int[] kernel_size;
        private int[] stride = {1, 1};
        private int[] padding = {0, 0};
        private String padding_mode = "zeros";
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

        public Builder padding_mode(String padding_mode) {
            if (padding_mode.equals("circular"))
                this.padding_mode = "circular";
            else
                this.padding_mode = "zeros";
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

        public Conv2d_N build() {
            return new Conv2d_N(this);
        }
    }
}
