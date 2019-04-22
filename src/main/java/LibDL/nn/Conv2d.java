package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Operator.BroadcastMul;
import LibDL.Tensor.Operator.Concat;
import LibDL.Tensor.Operator.Reshape;
import LibDL.Tensor.Operator.Sum;
import LibDL.Tensor.Operator.Unfold;
import LibDL.Tensor.Tensor;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

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

    private int filter_h;
    private int filter_w;
    private int _filter_h;
    private int _filter_w;
    private int amount_h;
    private int amount_w;

    public Conv2d(@NotNull Builder builder) {
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
            for(int j = 0; j < out_channels / groups; j++) {
                INDArray w = Nd4j.rand(new int[] {in_channels / groups, kernel_size[0], kernel_size[1]}).subi(0.5);
                zeros.put(new INDArrayIndex[] {NDArrayIndex.point(i*out_channels/groups+j),
                        NDArrayIndex.interval(i*in_channels/groups, (i+1)*in_channels/groups),
                        NDArrayIndex.all(), NDArrayIndex.all()}, w);
            }
        }
        INDArray temp = Nd4j.create(new double[][][][]{{{
                { 0.0533,  0.0976, -0.2318},
                  {-0.1049,  0.2347,  0.1320},
                  { 0.0219,  0.0274, -0.0140}},
        
                 {{-0.1144, -0.1571,  0.1365},
                  { 0.0889, -0.1669, -0.0461},
                  {-0.0925,  0.1974, -0.0859}}},
        
        
                {{{-0.0457,  0.0229, -0.1809},
                  {-0.1974, -0.1642,  0.1741},
                  {-0.0991,  0.0556,  0.1097}},
        
                 {{-0.0104,  0.0817,  0.0143},
                  { 0.1915,  0.0848, -0.1008},
                  { 0.2282, -0.1379,  0.0642}}}});
        temp = Nd4j.create(new double[][][][]{
                {{{ 0.2306,  0.1638, -0.1984},
          { 0.1396, -0.3312, -0.0261},
          { 0.2973,  0.0052,  0.0852}},
                        {{ 0,  0., -0.},
                                { 0., -0., -0.0},
                                { 0.,  0.00,  0.0}}
                },


        {{{ 0.,  0., -0},
                { 0., -0., -0.0},
                { 0.,  0.,  0.0}}, {{ 0.3048, -0.1866,  0.2185},
          { 0.1430, -0.2820,  0.2468},
          {-0.2531,  0.2290,  0.1932}}}});
        assert Arrays.equals(temp.shape(), zeros.shape());
        W = new Constant(temp.reshape(out_channels, in_channels*kernel_size[0]*kernel_size[1], 1), true);
        if(bias) {
            B = new Constant(Nd4j.rand(new int[] {out_channels}).subi(0.5), true);
        }else {
            B = null;
        }
    }
    public void W() {
        INDArray R = W.value.reshape(out_channels, in_channels*kernel_size[0]*kernel_size[1], 1).broadcast(out_channels, in_channels*kernel_size[0]*kernel_size[1], 9);
        System.out.println(W.value);
        System.out.println(R);
    }
//    private int getCounts() {
//        int filter_h = kernel_size[0];
//        int filter_w = kernel_size[1];
//        int _filter_h = (filter_h - 1) * dilation[0] + 1;
//        int _filter_w = (filter_w - 1) * dilation[1] + 1;
//        int amount_h = (input_h - _filter_h) / stride[0] + 1;
//        int amount_w = (input_w - _filter_w) / stride[1] + 1;
//    }

    @Override
    protected Tensor core() {
//        filter_h = kernel_size[0];
//        filter_w = kernel_size[1];
//        _filter_h = (filter_h - 1) * dilation[0] + 1;
//        _filter_w = (filter_w - 1) * dilation[1] + 1;
//        amount_h = (input_h - _filter_h) / stride[0] + 1;
//        amount_w = (input_w - _filter_w) / stride[1] + 1;
        Unfold unfold = new Unfold.Builder(input, kernel_size)
                .padding(padding)
                .stride(stride)
                .dilation(dilation)
                .build();
        BroadcastMul broadcastMul = new BroadcastMul(
                new Concat(unfold, out_channels, 1),
                new Reshape(W, 1, in_channels*out_channels*kernel_size[0]*kernel_size[1], 1));
        return new Sum(new Reshape(broadcastMul, true, out_channels, 5, 5), 2);
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
            if(kernel_size.length == 1) {
                this.kernel_size = new int[] {kernel_size[0], kernel_size[0]};
            }else {
                this.kernel_size = kernel_size;
            }
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
