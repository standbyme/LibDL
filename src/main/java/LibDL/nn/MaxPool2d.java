package LibDL.nn;

import LibDL.Tensor.Operator.Padding;
import LibDL.Tensor.Operator.Unfold;
import LibDL.Tensor.Tensor;

public class MaxPool2d extends Module {

    private int[] kernel_size;
    private int[] stride;
    private int[] padding;
    private int[] dilation;
    private boolean return_indices;
    private boolean ceil_mode;

    private MaxPool2d(Builder builder) {
        kernel_size = builder.kernel_size;
        stride = builder.stride;
        padding = builder.padding;
        dilation = builder.dilation;
        return_indices = builder.return_indices;
        ceil_mode = builder.ceil_mode;
    }

    @Override
    public Tensor forward(Tensor input) {
        return new Unfold.Builder(new Padding(input, kernel_size, padding, stride, dilation, ceil_mode), kernel_size)
                .stride(stride)
                .dilation(dilation)
                .padding(0) // required
                .build();
    }

    public static class Builder {
        private int[] kernel_size;
        private int[] stride;
        private int[] padding = {0, 0};
        private int[] dilation = {1, 1};
        private boolean return_indices = false;
        private boolean ceil_mode = false;

        public Builder(int... kernel_size) {
            if(kernel_size.length == 1) {
                this.kernel_size = new int[] {kernel_size[0], kernel_size[0]};
            }else {
                this.kernel_size = kernel_size;
            }
            this.stride = kernel_size;
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
        public Builder return_indices(boolean return_indices) {
            this.return_indices = return_indices;
            return this;
        }
        public Builder ceil_mode(boolean ceil_mode) {
            this.ceil_mode = ceil_mode;
            return this;
        }
        public MaxPool2d build() {
            return new MaxPool2d(this);
        }
    }
}
