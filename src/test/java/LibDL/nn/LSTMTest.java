package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Parameter;
import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

public class LSTMTest {

    private static LSTM lstm = null;

    @BeforeClass
    public static void initRNN() {
        lstm = new LSTM(2, 1, 1);

        lstm.setParam(LSTM.WEIGHT_IH, Nd4j.create(
                new double[][]{
                        {1, 1},
                        {0, 0},
                        {3.5, 3.5},
                        {2, 0}
                }
        ));
        lstm.setParam(LSTM.WEIGHT_HH, Nd4j.create(
                new double[][]{
                        {1},
                        {1},
                        {1},
                        {2}
                }
        ));
        lstm.setParam(LSTM.BIAS_IH, Nd4j.create(new double[]{
                1, 0, 3, 0
        }).transpose());
        lstm.setParam(LSTM.BIAS_HH, Nd4j.create(new double[]{
                1, 2, 0, 2
        }).transpose());
    }

    @Test
    public void test() {

        Parameter[] parameters = lstm.parameters();

        assert parameters.length == 16;

        Variable input = new Variable(Nd4j.create(new double[][][]{
                {{1, 1},
                        {0.5, 0.4},
                        {1, 0.1}},
                {{1, 1.9},
                        {0, 1.9},
                        {1, 1.9}}}), true);
        Constant h0 = new Constant(Nd4j.create(new double[][]{
                {1.5},
                {1.6},
                {1.1}}));
        Constant c0 = new Constant(Nd4j.create(new double[][]{
                {0.5},
                {-1.6},
                {10.1}}));

        Tensor result = lstm.forward(input, h0, c0);
        System.out.println(result);
        assert result.data.equalsWithEps(Nd4j.create(new double[][][]
                {{{0.9009},
                        {-0.5132},
                        {0.9980}},

                        {{0.9807},
                                {0.3375},
                                {0.9975}}}), 1e-3);
        System.out.println(lstm.h_n[0].data);
        assert lstm.h_n[0].data.equalsWithEps(Nd4j.create(new double[]
                {0.9807, 0.3375, 0.9975}
        ), 1e-3);

        assert lstm.c_n[0].data.equalsWithEps(Nd4j.create(new double[]{
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


}
