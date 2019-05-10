package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Parameter;
import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

public class GRUTest {
    private static GRU gru = null;

    @BeforeClass
    public static void initRNN() {

        gru = new GRU(2, 1);

        gru.setParam(Nd4j.create(
                new double[][]{
                        {1, 1},
                        {0, 0},
                        {3.5, 3.5}
                }
        ), GRU.WEIGHT_IH);
        gru.setParam(Nd4j.create(
                new double[][]{
                        {1},
                        {1},
                        {1}
                }
        ), GRU.WEIGHT_HH);
        gru.setParam(Nd4j.create(new double[]{
                1, 0, 3
        }).transpose(), GRU.BIAS_IH);
        gru.setParam(Nd4j.create(new double[]{
                1, 2, 0
        }).transpose(), GRU.BIAS_HH);

    }

    @Test
    public void testGRU() {
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
        Parameter[] parameters = gru.parameters();

        assert parameters.length == 12;

        Tensor result = gru.forward(input, h0);

        assert result.data.equalsWithEps(Nd4j.create(
                new double[][][]
                        {{{1.4853},
                                {1.5840},
                                {1.0957}},

                                {{1.4709},
                                        {1.5683},
                                        {1.0915}}}
        ), 1e-3);

        System.out.println(result);
        result.sum().backward();
        System.out.println(input.grad);

        assert input.grad.equalsWithEps(Nd4j.create(new double[][][]
                {{{0.0000e+00, 0.0000e+00},
                        {1.3305e-07, 1.3305e-07},
                        {1.4170e-07, 1.4170e-07}},

                        {{0.0000e+00, 0.0000e+00},
                                {0.0000e+00, 0.0000e+00},
                                {0.0000e+00, 0.0000e+00}}}), 1e-3);
    }

}
