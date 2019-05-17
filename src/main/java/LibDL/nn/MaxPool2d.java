package LibDL.nn;

import LibDL.Tensor.Operator.Max;
import LibDL.Tensor.Operator.Padding;
import LibDL.Tensor.Operator.Reshape;
import LibDL.Tensor.Operator.Unfold;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.OldFModOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

public class MaxPool2d extends Module {

    private int[] kernel_size;
    private int[] stride;
    private int[] padding;
    private int[] dilation;
    private boolean return_indices;
    private boolean ceil_mode;

    private INDArray indices;

    private MaxPool2d(Builder builder) {
        kernel_size = builder.kernel_size;
        stride = builder.stride;
        padding = builder.padding;
        dilation = builder.dilation;
        return_indices = builder.return_indices;
        ceil_mode = builder.ceil_mode;
    }

    public MaxPool2d(int[] kernel_size,
                     int[] stride, int[] padding, int[] dilation, boolean return_indices, boolean ceil_mode) {
        this.kernel_size = kernel_size;
        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        this.return_indices = return_indices;
        this.ceil_mode = ceil_mode;

        if(stride == null) {
            this.stride = kernel_size;
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        long[] shape = input.data.shape();
        long[] to_shape_1 = new long[]{
                input.data.rank() == 4 ? shape[0] * shape[1] : shape[0],
                1, shape[input.data.rank() - 2], shape[input.data.rank() - 1]
        };

        Padding padding = new Padding(
                new Reshape(input, to_shape_1), kernel_size, this.padding, stride, dilation, ceil_mode);
        Unfold unfold = new Unfold.Builder(padding, kernel_size)
                .stride(stride)
                .dilation(dilation)
                .padding(0) // required
                .build();
        Max max = new Max(unfold, 1);

        long[] to_shape_2 = Arrays.copyOf(shape, shape.length);
        long amount_h = unfold.getAmount()[0];
        long amount_w = unfold.getAmount()[1];
        to_shape_2[input.data.rank() - 2] = amount_h;
        to_shape_2[input.data.rank() - 1] = amount_w;

        if(return_indices) {
            to_shape_1[2] = amount_h;
            to_shape_1[3] = amount_w;

            indices = max.getArgMax().reshape(to_shape_1);

            long input_h = shape[input.data.rank() - 2];
            long input_w = shape[input.data.rank() - 1];

            INDArrayIndex[] indArrayIndices;
            INDArray n, row, col, mod = Nd4j.zeros(to_shape_1[0], to_shape_1[1]);
            INDArray ks1 = Nd4j.onesLike(mod).muli(kernel_size[1]);
            INDArray ih = Nd4j.onesLike(mod).muli(input_h);
            INDArray iw = Nd4j.onesLike(mod).muli(input_w);
            INDArray mod_row_col = Nd4j.zerosLike(mod);
            for (int i = 0; i < amount_h; i++) {
                for (int j = 0; j < amount_w; j++) {
//                    math:
//                    row = i * stride[0] + n / filter_w * dilation[0];
//                    col = j * stride[1] + n % filter_w * dilation[1];
                    indArrayIndices = new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(),
                            NDArrayIndex.point(i), NDArrayIndex.point(j)};
                    n = indices.get(indArrayIndices);
                    Nd4j.getExecutioner().execAndReturn(new OldFModOp(n, ks1, mod));
                    col = mod.mul(dilation[1]);
                    col.addi(j * stride[1]);
                    row = n.subi(mod).divi(ks1).muli(dilation[0]).addi(i * stride[0]);
                    row.subi(this.padding[0]);
                    col.subi(this.padding[1]);

                    Nd4j.getExecutioner().exec(new OldFModOp(row, ih, mod_row_col));
                    row.addi(row.sub(mod_row_col).mul(-2));
                    Nd4j.getExecutioner().exec(new OldFModOp(col, iw, mod_row_col));
                    col.addi(col.sub(mod_row_col).mul(-input_h - 2));

                    n = row.muli(input_w).addi(col);

                    n.addi(1).addi(Transforms.abs(n)).divi(2).subi(1);

                    indices.put(indArrayIndices, n).shape();
                }
            }
            indices = indices.reshape(to_shape_2);
        }

        return new Reshape(max, to_shape_2);
    }

    public INDArray getIndices() {
        if (return_indices)
            return indices;
        return null;
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
