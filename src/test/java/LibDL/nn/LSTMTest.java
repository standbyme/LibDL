package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Parameter;
import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class LSTMTest {

//    private static LSTM test_lstm = null;
//
//    @BeforeClass
//    public static void initRNN() {
//    }

    @Test
    public void test() {
        LSTM test_lstm = new LSTM(2, 1, 1);

        test_lstm.setParam(LSTM.WEIGHT_IH, Nd4j.create(
                new double[][]{
                        {1, 1},
                        {0, 0},
                        {3.5, 3.5},
                        {2, 0}
                }
        ));
        test_lstm.setParam(LSTM.WEIGHT_HH, Nd4j.create(
                new double[][]{
                        {1},
                        {1},
                        {1},
                        {2}
                }
        ));
        test_lstm.setParam(LSTM.BIAS_IH, Nd4j.create(new double[]{
                1, 0, 3, 0
        }).transpose());
        test_lstm.setParam(LSTM.BIAS_HH, Nd4j.create(new double[]{
                1, 2, 0, 2
        }).transpose());
        Parameter[] parameters = test_lstm.parameters();

        assert parameters.length == 16;

        Variable input = new Variable(Nd4j.create(new double[][][]{
                {{1, 1},
                        {0.5, 0.4},
                        {1, 0.1}},
                {{1, 1.9},
                        {0, 1.9},
                        {1, 1.9}}}), true);
        Constant h0 = new Constant(Nd4j.create(new double[][][]{{
                {1.5},
                {1.6},
                {1.1}}}));
        Constant c0 = new Constant(Nd4j.create(new double[][][]{{
                {0.5},
                {-1.6},
                {10.1}}}));

        Tensor result = test_lstm.forward(input, h0, c0);
        System.out.println(result);
        assert result.data.equalsWithEps(Nd4j.create(new double[][][]
                {{{0.9009},
                        {-0.5132},
                        {0.9980}},

                        {{0.9807},
                                {0.3375},
                                {0.9975}}}), 1e-3);
        System.out.println(test_lstm.h_n[0].data);
        assert test_lstm.h_n[0].data.equalsWithEps(Nd4j.create(new double[]
                {0.9807, 0.3375, 0.9975}
        ), 1e-3);

        assert test_lstm.c_n[0].data.equalsWithEps(Nd4j.create(new double[]{
                2.4011, 0.5037, 11.1411
        }), 1e-3);
        result.sum().backward();
        System.out.println(input.grad);
        input.grad.equalsWithEps(Nd4j.create(new double[][][]
                {{{2.5423e-03, 8.8714e-04},
                        {1.1854e-02, 1.4254e-02},
                        {4.0624e-03, 0.0000e+00}},

                        {{6.0070e-03, 9.6868e-05},
                                {2.0306e-01, 1.8000e-02},
                                {4.9530e-03, 0.0000e+00}}}
        ), 1e-3);

    }

    @Test
    public void testShape() {
        try {
            INDArray input = Nd4j.rand(new int[]{12, 50, 100});
            LSTM ls = new LSTM(100, 200, 1);
            Tensor out = ls.forward(new Variable(input));
            System.out.println(Arrays.toString(out.sizes()));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    @Test
    public void testMultipleLayer() {

        LSTM lstm = new LSTM(2, 1, 3);
        lstm.setParam(LSTM.WEIGHT_IH,
                Nd4j.create(new double[][]{{1., 1.},
                        {0, 0},
                        {3.5, 3.5},
                        {2, 0}}
                )//layer 0
                , Nd4j.create(new double[][]{{1.},
                        {0},
                        {3.5},
                        {2.0}}
                )//layer 1
                , Nd4j.create(new double[][]{{1.},
                        {0},
                        {3.5},
                        {2.0}}
                )//layer 2
        );
        lstm.setParam(RNNBase.WEIGHT_HH,
                Nd4j.create(new double[][]{{1.},
                        {1.},
                        {1.},
                        {2.}}),
                Nd4j.create(new double[][]{{1.},
                        {1.},
                        {1.},
                        {2.}}
                ),
                Nd4j.create(new double[][]{{1.},
                        {1.},
                        {1.},
                        {2.}}
                )
        );
        lstm.setParam(RNNBase.BIAS_IH,
                Nd4j.create(new double[]{1., 0, 3, 0}).transpose(),
                Nd4j.create(new double[]{1., 0, 3, 0}).transpose(),
                Nd4j.create(new double[]{1., 0, 3, 0}).transpose()
        );
        lstm.setParam(RNNBase.BIAS_HH,
                Nd4j.create(new double[]{1., 2, 0, 2}).transpose(),
                Nd4j.create(new double[]{1., 2, 0, 2}).transpose(),
                Nd4j.create(new double[]{1., 2, 0, 2}).transpose()
        );

        Tensor data = new Variable(Nd4j.create(new double[][][]{
                {{1., 1.},
                        {0.5, 0.4},
                        {1., 0.1}},

                {{1., 1.9},
                        {0., 1.9},
                        {1., 1.9}}}), true);
        Tensor h0 = new Variable(Nd4j.create(new double[][][]{
                {{1.5},
                        {1.6},
                        {1.1}},
                {{1.25},
                        {0.6},
                        {1.1}},
                {{0.5},
                        {1.6},
                        {0.1}}}), true);
        Tensor c0 = new Variable(Nd4j.create(new double[][][]{
                {{0.5},
                        {-1.6},
                        {10.1}},
                {{-9.5},
                        {-11.6},
                        {1.1}},
                {{0.25},
                        {-1.56},
                        {5.1}}}), true);
        Tensor output = lstm.forward(data, h0, c0);
        System.out.println(output);
        output.data.equalsWithEps(Nd4j.create(new double[][]{
                {0.1698,
                        -0.5711,
                        0.9841},

                {0.1815,
                        -0.1897,
                        0.9974}
        }), 0.0003);
        Tensor hn = lstm.hn();
        hn.data.equalsWithEps(Nd4j.create(new double[][][]{{{0.9807},
                {0.3375},
                {0.9975}},

                {{-0.8770},
                        {-0.7067},
                        {0.9915}},

                {{0.1815},
                        {-0.1897},
                        {0.9974}}}
        ), 0.003);

        Tensor cn = lstm.cn();

        cn.data.equalsWithEps(Nd4j.create(new double[][][]{{{2.4011},
                {0.5037},
                {11.1411}},

                {{-5.0901},
                        {-6.6670},
                        {2.9177}},

                {{0.2904},
                        {-0.5766},
                        {6.2158}}}), 0.003);

    }
}