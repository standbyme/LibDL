package LibDL.nn;

import LibDL.Tensor.Operator.*;
import LibDL.Tensor.Operator.Reshape;
import LibDL.Tensor.Parameter;
import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.Assert.assertEquals;

public class Conv2dTest {

    class Conv extends Conv2d {

        private int in_channels;
        private int out_channels;
        private int[] kernel_size;
        private int[] stride;
        private int[] padding;
        private String padding_mode;
        private int[] dilation;
        private int groups;
        private boolean bias;

        INDArray data;
        INDArray grad;
        Tensor core;

        private Parameter W;
        private Parameter B;

        Conv(int in_channels, int out_channels, int[] kernel_size, int[] stride, int[] padding, String padding_mode, int[] dilation, int groups, boolean bias) {
            super(in_channels, out_channels, kernel_size, stride, padding, padding_mode, dilation, groups, bias);
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
        void backward() {
            core.grad = grad;
            core.backward();
        }

        void apply(Tensor input) {
            core = forward(input);
            data = core.data;
        }

        void setB(INDArray value) {
            B = new Parameter(value);
        }

        void setW(INDArray value) {
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
                        (padding[1] + 1) / 2, padding[1] / 2, (padding[0] + 1) / 2, padding[0] / 2);
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
            BroadcastMul broadcastMul = new BroadcastMul(
                    new Concat(unfold, out_channels, 1),
                    new LibDL.Tensor.Operator.Reshape(W, 1, in_channels * out_channels * kernel_size[0] * kernel_size[1], 1),
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
    }

    @Test
    public void testConv2d() {
        Variable input = new Variable(Nd4j.linspace(1, 192, 192).reshape(2, 2, 8, 6), true);
        Conv conv2d = new Conv(2, 4, new int[]{3, 2},
                new int[]{2, 1}, new int[]{1, 1}, "zeros", new int[]{1, 2}, 2, true);
        conv2d.setW(Nd4j.create(new double[][][][] {
        {{{-0.14339730144,  0.11298561096},
          {-0.30034396052, -0.20663771033},
          {-0.39040639997,  0.15988129377}}},


        {{{-0.02674335241,  0.28867983818},
          { 0.15186113119,  0.39023303986},
          { 0.36364132166, -0.00603255630}}},


        {{{-0.23355589807, -0.32926410437},
          { 0.16129279137,  0.02150997519},
          { 0.25931823254,  0.04973560572}}},


        {{{-0.38155251741, -0.30007559061},
          {-0.01913371682,  0.40136158466},
          { 0.20428019762, -0.28288879991}}}}));
        conv2d.setB(Nd4j.create(new double[]
                {-0.19259913266, -0.28515064716, -0.29368337989,  0.33625578880}).reshape(4));
        conv2d.apply(input);
        INDArray expected = Nd4j.create(new double[][][][]{
        {{{ 6.73175811768e-01, -2.40676927567e+00, -3.14427614212e+00, -3.88178300858e+00, -4.61928939819e+00, -5.98878955841e+00},
          { 1.01598370075e+00, -1.12437620163e+01, -1.20116796494e+01, -1.27795982361e+01, -1.35475168228e+01, -1.58551645279e+01},
          { 1.81073391438e+00, -2.04587841034e+01, -2.12267017365e+01, -2.19946193695e+01, -2.27625370026e+01, -2.58649349213e+01},
          { 2.60548424721e+00, -2.96738052368e+01, -3.04417209625e+01, -3.12096443176e+01, -3.19775600433e+01, -3.58747062683e+01}},

         {{ 4.47054982185e-01,  3.52860593796e+00,  4.42830896378e+00,  5.32801151276e+00,  6.22771453857e+00,  4.47420930862e+00},
          { 7.36689949036e+00,  1.67359561920e+01,  1.78975963593e+01,  1.90592346191e+01,  2.02208728790e+01,  1.03660621643e+01},
          { 1.54414634705e+01,  3.06756286621e+01,  3.18372707367e+01,  3.29989051819e+01,  3.41605453491e+01,  1.62311725616e+01},
          { 2.35160274506e+01,  4.46152992249e+01,  4.57769393921e+01,  4.69385795593e+01,  4.81002159119e+01,  2.20962810516e+01}},

         {{ 3.56700944901e+00,  2.58041038513e+01,  2.62959594727e+01,  2.67878170013e+01,  2.72796726227e+01,  2.35546092987e+01},
          {-1.40168333054e+01,  9.27568972111e-02,  2.17920839787e-02, -4.91727292538e-02, -1.20133727789e-01,  1.48221454620e+01},
          {-1.71130580902e+01, -7.58803725243e-01, -8.29768538475e-01, -9.00735259056e-01, -9.71696257591e-01,  1.70668048859e+01},
          {-2.02092800140e+01, -1.61036622524e+00, -1.68133103848e+00, -1.75229203701e+00, -1.82325494289e+00,  1.93114681244e+01}},

         {{ 4.56256198883e+00,  1.49788923264e+01,  1.52825126648e+01,  1.55861320496e+01,  1.58897514343e+01,  1.13747005463e+01},
          {-1.08199977875e+01, -1.94673709869e+01, -1.98453807831e+01, -2.02233867645e+01, -2.06014003754e+01, -8.91514110565e+00},
          {-1.29992294312e+01, -2.40034751892e+01, -2.43814888000e+01, -2.47594947815e+01, -2.51375026703e+01, -1.12720146179e+01},
          {-1.51784629822e+01, -2.85395851135e+01, -2.89175930023e+01, -2.92956047058e+01, -2.96736106873e+01, -1.36288871765e+01}}},


        {{{-3.81543993950e+00, -7.32074127197e+01, -7.39449234009e+01, -7.46824340820e+01, -7.54199371338e+01, -7.23008193970e+01},
          { 7.37398624420e+00, -8.49639358521e+01, -8.57318496704e+01, -8.64997711182e+01, -8.72676849365e+01, -9.59333343506e+01},
          { 8.16873645782e+00, -9.41789474487e+01, -9.49468688965e+01, -9.57147827148e+01, -9.64827117920e+01, -1.05943115234e+02},
          { 8.96348667145e+00, -1.03393974304e+02, -1.04161895752e+02, -1.04929809570e+02, -1.05697723389e+02, -1.15952880859e+02}},

         {{ 3.73302993774e+01,  8.99000930786e+01,  9.07997970581e+01,  9.16994934082e+01,  9.25992050171e+01,  5.39624443054e+01},
          { 7.19634170532e+01,  1.28253326416e+02,  1.29414978027e+02,  1.30576614380e+02,  1.31738250732e+02,  5.72869300842e+01},
          { 8.00379714966e+01,  1.42193008423e+02,  1.43354644775e+02,  1.44516281128e+02,  1.45677932739e+02,  6.31520423889e+01},
          { 8.81125411987e+01,  1.56132675171e+02,  1.57294311523e+02,  1.58455963135e+02,  1.59617599487e+02,  6.90171585083e+01}},

         {{ 1.04065856934e+01,  7.30223388672e+01,  7.35141906738e+01,  7.40060501099e+01,  7.44979095459e+01,  6.39332695007e+01},
          {-3.87866134644e+01, -6.71973180771e+00, -6.79069280624e+00, -6.86165380478e+00, -6.93262243271e+00,  3.27794380188e+01},
          {-4.18828315735e+01, -7.57128667831e+00, -7.64225530624e+00, -7.71321630478e+00, -7.78418111801e+00,  3.50241012573e+01},
          {-4.49790534973e+01, -8.42286014557e+00, -8.49381732941e+00, -8.56478214264e+00, -8.63574314117e+00,  3.72687568665e+01}},

         {{ 1.59359531403e+01,  4.41263389587e+01,  4.44299621582e+01,  4.47335777283e+01,  4.50372009277e+01,  2.91487636566e+01},
          {-2.82538661957e+01, -5.57562217712e+01, -5.61342353821e+01, -5.65122337341e+01, -5.68902473450e+01, -2.77701206207e+01},
          {-3.04331035614e+01, -6.02923316956e+01, -6.06703376770e+01, -6.10483436584e+01, -6.14263572693e+01, -3.01269931793e+01},
          {-3.26123390198e+01, -6.48284301758e+01, -6.52064361572e+01, -6.55844573975e+01, -6.59624710083e+01, -3.24838600159e+01}}}});
        assertEquals(expected, conv2d.data);
        conv2d.grad = Nd4j.ones(conv2d.data.shape());
        conv2d.backward();
        assertEquals(Nd4j.create(new double[][][][]{
        {{{-0.14848282933,  0.03511250019,  0.03511250019,  0.03511250019,  0.03511250019,  0.18359532952},
          {-0.19690573215,  0.35860845447,  0.35860845447,  0.35860845447,  0.35860845447,  0.55551421642},
          {-0.14848282933,  0.03511250019,  0.03511250019,  0.03511250019,  0.03511250019,  0.18359532952},
          {-0.19690573215,  0.35860845447,  0.35860845447,  0.35860845447,  0.35860845447,  0.55551421642},
          {-0.14848282933,  0.03511250019,  0.03511250019,  0.03511250019,  0.03511250019,  0.18359532952},
          {-0.19690573215,  0.35860845447,  0.35860845447,  0.35860845447,  0.35860845447,  0.55551421642},
          {-0.14848282933,  0.03511250019,  0.03511250019,  0.03511250019,  0.03511250019,  0.18359532952},
          {-0.02676507831,  0.12708365917,  0.12708365917,  0.12708365917,  0.12708365917,  0.15384873748}},

         {{ 0.14215907454,  0.56503063440,  0.56503063440,  0.56503063440,  0.56503063440,  0.42287155986},
          {-0.15151000023, -1.01400291920, -1.01400291920, -1.01400291920, -1.01400291920, -0.86249291897},
          { 0.14215907454,  0.56503063440,  0.56503063440,  0.56503063440,  0.56503063440,  0.42287155986},
          {-0.15151000023, -1.01400291920, -1.01400291920, -1.01400291920, -1.01400291920, -0.86249291897},
          { 0.14215907454,  0.56503063440,  0.56503063440,  0.56503063440,  0.56503063440,  0.42287155986},
          {-0.15151000023, -1.01400291920, -1.01400291920, -1.01400291920, -1.01400291920, -0.86249291897},
          { 0.14215907454,  0.56503063440,  0.56503063440,  0.56503063440,  0.56503063440,  0.42287155986},
          { 0.46359843016,  0.23044523597,  0.23044523597,  0.23044523597,  0.23044523597, -0.23315319419}}},


        {{{-0.14848282933,  0.03511250019,  0.03511250019,  0.03511250019,  0.03511250019,  0.18359532952},
          {-0.19690573215,  0.35860845447,  0.35860845447,  0.35860845447,  0.35860845447,  0.55551421642},
          {-0.14848282933,  0.03511250019,  0.03511250019,  0.03511250019,  0.03511250019,  0.18359532952},
          {-0.19690573215,  0.35860845447,  0.35860845447,  0.35860845447,  0.35860845447,  0.55551421642},
          {-0.14848282933,  0.03511250019,  0.03511250019,  0.03511250019,  0.03511250019,  0.18359532952},
          {-0.19690573215,  0.35860845447,  0.35860845447,  0.35860845447,  0.35860845447,  0.55551421642},
          {-0.14848282933,  0.03511250019,  0.03511250019,  0.03511250019,  0.03511250019,  0.18359532952},
          {-0.02676507831,  0.12708365917,  0.12708365917,  0.12708365917,  0.12708365917,  0.15384873748}},

         {{ 0.14215907454,  0.56503063440,  0.56503063440,  0.56503063440,  0.56503063440,  0.42287155986},
          {-0.15151000023, -1.01400291920, -1.01400291920, -1.01400291920, -1.01400291920, -0.86249291897},
          { 0.14215907454,  0.56503063440,  0.56503063440,  0.56503063440,  0.56503063440,  0.42287155986},
          {-0.15151000023, -1.01400291920, -1.01400291920, -1.01400291920, -1.01400291920, -0.86249291897},
          { 0.14215907454,  0.56503063440,  0.56503063440,  0.56503063440,  0.56503063440,  0.42287155986},
          {-0.15151000023, -1.01400291920, -1.01400291920, -1.01400291920, -1.01400291920, -0.86249291897},
          { 0.14215907454,  0.56503063440,  0.56503063440,  0.56503063440,  0.56503063440,  0.42287155986},
          { 0.46359843016,  0.23044523597,  0.23044523597,  0.23044523597,  0.23044523597, -0.23315319419}}}}),
                input.grad);
        assertEquals(Nd4j.create(new double[][][][]{
                 {{{ 2070.0000, 2100.0000},
                   { 2760.0000, 2800.0000},
                   { 3000.0000, 3040.0000}},

                  {{         0,         0},
                   {         0,         0},
                   {         0,         0}}},


                 {{{ 2070.0000, 2100.0000},
                   { 2760.0000, 2800.0000},
                   { 3000.0000, 3040.0000}},

                  {{         0,         0},
                   {         0,         0},
                   {         0,         0}}},


                 {{{         0,         0},
                   {         0,         0},
                   {         0,         0}},

                  {{ 3510.0000, 3540.0000},
                   { 4680.0000, 4720.0000},
                   { 4920.0000, 4960.0000}}},


                 {{{         0,         0},
                   {         0,         0},
                   {         0,         0}},

                  {{ 3510.0000, 3540.0000},
                   { 4680.0000, 4720.0000},
                   { 4920.0000, 4960.0000}}}}), conv2d.getW().grad);
        assertEquals(Nd4j.create(new double[]{48., 48., 48., 48.}), conv2d.getB().grad);

    }
}
