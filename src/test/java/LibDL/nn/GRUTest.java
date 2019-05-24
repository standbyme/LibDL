package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Parameter;
import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

public class GRUTest {
    private static GRU gru = null;

    @BeforeClass
    public static void initRNN() {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        gru = new GRU(2, 1, 1);

        gru.setParam(GRU.WEIGHT_IH, Nd4j.create(
                new double[][]{
                        {1, 1},
                        {0, 0},
                        {3.5, 3.5}
                }
        ));
        gru.setParam(GRU.WEIGHT_HH, Nd4j.create(
                new double[][]{
                        {1},
                        {1},
                        {1}
                }
        ));
        gru.setParam(GRU.BIAS_IH, Nd4j.create(new double[]{
                1, 0, 3
        }));
        gru.setParam(GRU.BIAS_HH, Nd4j.create(new double[]{
                1, 2, 0
        }));

    }

    @Test
    public void test() {
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
        System.out.println(result);

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
