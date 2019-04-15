package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Operator.Unfold;
import LibDL.Tensor.Tensor;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.factory.Nd4j;

public class Conv2d extends LayerTensor {

    private int in_channels;
    private int out_channels;
    private int[] kernel_size;
    private int[] stride;
    private int[] padding;
    private int[] dilation;
    private int groups;
    private boolean bias;

    private final Constant W;
    private final Constant B;

    public Conv2d(@NotNull Builder builder) {
        this.in_channels = builder.in_channels;
        this.out_channels = builder.out_channels;
        this.kernel_size = builder.kernel_size;
        this.stride = builder.stride;
        this.padding = builder.padding;
        dilation = builder.dilation;
        groups = builder.groups;
        bias = builder.bias;

        W = new Constant(Nd4j.rand(new int[] {out_channels, in_channels, kernel_size[0], kernel_size[1]}), true);
        if(bias) {
            B = new Constant(Nd4j.rand(new int[] {out_channels}), true);
        }else {
            B = null;
        }
    }

    @Override
    protected Tensor core() {

        Unfold col = new Unfold.Builder(input, 3, 3).padding(2, 2).build();

        if (bias) return col.mm(W).add(B).reshapeLike(input);
        else return col.mm(W).reshapeLike(input);

    }

    public class Builder {
        private int in_channels;
        private int out_channels;
        private int[] kernel_size;
        private int[] stride = {1, 1};
        private int[] padding = {0, 0};
        private int[] dilation = {1, 1};
        private int groups = 1;
        private boolean bias = true;
        public Builder(int in_channels, int out_channels, int kernel_size) {
            this.in_channels = in_channels;
            this.out_channels = out_channels;
            this.kernel_size = new int[] {kernel_size, kernel_size};
        }
        public Builder(int in_channels, int out_channels, int... kernel_size) {
            this.in_channels = in_channels;
            this.out_channels = out_channels;
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
