package LibDL.Tensor.Operator;

import LibDL.Tensor.Constant;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class main {

    @Test
    public void testMax() {
        Constant data = new Constant(Nd4j.create(new double[][]{
                {0.3, 4.0, 2.9},
                {3.5, 2.2, 2.5},
                {0.5, 6, 6.5},
                {0.5, 6, 7.5},

        }), true);

        Max result = new Max(data);

        result.forward();

        assertEquals(result.out, Nd4j.create(new double[]{4.0, 3.5, 6.5, 7.5}));

        result.dout = Nd4j.create(new double[]{0.5, 1, 2, 4});

        result.backward();
        assertEquals(data.dout, Nd4j.create(new double[][]{
                {0, 0.5, 0},
                {1, 0, 0},
                {0, 0, 2},
                {0, 0, 4}
        }));

    }

    @Test
    public void testCrossEntropyLoss() {
        Constant t = new Constant(Nd4j.create(new double[]{0.0, 0, 1, 0, 0, 0, 0, 0, 0, 0}));
        Constant y = new Constant(Nd4j.create(new double[]{0.1, 0.05, 0.6, 0, 0.05, 0.1, 0, 0.1, 0, 0}));


        CrossEntropyLoss result = new CrossEntropyLoss(y, t);
        result.forward();
        double a = result.out.getDouble(0);

        assert a == 0.5108253955841064;

        y = new Constant(Nd4j.create(new double[]{0.1, 0.05, 0.1, 0, 0.05, 0.1, 0, 0.6, 0, 0}));
        result = new CrossEntropyLoss(y, t);
        result.forward();

        double b = result.out.getDouble(0);

        assert b == 2.302584171295166;

    }

    @Test
    public void testUnfold() {

        Constant x = new Constant(Nd4j.linspace(0, 15, 16).reshape(4, 4), true);

        Unfold ret = new Unfold(x, 3, 0, 1);
        ret.forward();

        INDArray assertion_forward = Nd4j.create(new double[][]{{0, 1, 2, 4, 5, 6, 8, 9, 10}, {1, 2, 3, 5, 6, 7, 9, 10, 11}, {4, 5, 6, 8, 9, 10, 12, 13, 14}, {5, 6, 7, 9, 10, 11, 13, 14, 15}});

        assertEquals(assertion_forward, ret.out);

        ret.dout = Nd4j.create(new double[][]{{0, 1, 2, 3, 0, 1, 2, 3, 0}, {0, 1, 2, 3, 0, 1, 2, 3, 0}, {0, 1, 2, 3, 0, 1, 2, 3, 0}, {0, 1, 2, 3, 0, 1, 2, 3, 0}});
        ret.backward();

        INDArray assertion_backward = Nd4j.create(new double[][]{{0, 1, 3, 2}, {3, 4, 4, 3}, {5, 8, 4, 1}, {2, 5, 3, 0}});

        assertEquals(assertion_backward, x.dout);
    }


    @Test
    public void testReshape() {

        INDArray matrix = Nd4j.linspace(0, 5, 6);
        INDArray matrix_2_3 = matrix.reshape(2, 3);
        INDArray matrix_3_2 = matrix.reshape(3, 2);

        Constant x = new Constant(matrix_2_3, true);
        Reshape reshape = new Reshape(x, 3, 2);

        reshape.forward();
        assertArrayEquals(new long[]{2, 3}, reshape.from_shape);

        reshape.dout = matrix_3_2;
        reshape.backward();
        assertEquals(matrix_2_3, x.dout);

    }

    @Test
    public void testAddAndSub() {
        Constant data1 = new Constant(Nd4j.create(new double[]{1, 2, 3}));
        Constant data2 = new Constant(Nd4j.create(new double[]{1, 2, 3}));

        Add add = new Add(data1, data2);
        Sub sub = new Sub(data1, data2);

        add.forward();
        sub.forward();

        {
            double a = add.out.getDouble(0);
            double b = add.out.getDouble(1);
            double c = add.out.getDouble(2);

            assert a == 2;
            assert b == 4;
            assert c == 6;
        }

        {
            double a = sub.out.getDouble(0);
            double b = sub.out.getDouble(1);
            double c = sub.out.getDouble(2);

            assert a == 0;
            assert b == 0;
            assert c == 0;
        }

    }

    @Test
    public void testAverage() {
        Constant data = new Constant(Nd4j.create(new double[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12},

        }), true);

        Average result = new Average(data);

        result.forward();

        assertEquals(result.out, Nd4j.create(new double[]{2, 5, 8, 11}));

        result.dout = Nd4j.create(new double[]{3, 6, 9, 12});

        result.backward();
        assertEquals(data.dout, Nd4j.create(new double[][]{
                {1, 1, 1},
                {2, 2, 2},
                {3, 3, 3},
                {4, 4, 4}
        }));
    }

}
