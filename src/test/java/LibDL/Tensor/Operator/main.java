package LibDL.Tensor.Operator;

import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class main {

    @Test
    public void testMax() {
        Variable data = new Variable(Nd4j.create(new double[][]{
                {0.3, 4.0, 2.9},
                {3.5, 2.2, 2.5},
                {0.5, 6, 6.5},
                {0.5, 6, 7.5},

        }), true);

        Max result = new Max(data);

        result.forwardWithInput();

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
    public void testReshape() {

        INDArray matrix = Nd4j.linspace(0, 5, 6);
        INDArray matrix_2_3 = matrix.reshape(2, 3);
        INDArray matrix_3_2 = matrix.reshape(3, 2);

        Variable x = new Variable(matrix_2_3, true);
        Reshape reshape = new Reshape(x, 3, 2);

        reshape.forwardWithInput();
        assertArrayEquals(new long[]{2, 3}, reshape.from_shape);

        reshape.dout = matrix_3_2;
        reshape.backward();
        assertEquals(matrix_2_3, x.dout);

    }

    @Test
    public void testAddAndSub() {
        Variable data1 = new Variable(Nd4j.create(new double[]{1, 2, 3}));
        Variable data2 = new Variable(Nd4j.create(new double[]{1, 2, 3}));

        Add add = new Add(data1, data2);
        Sub sub = new Sub(data1, data2);

        add.forwardWithInput();
        sub.forwardWithInput();

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
        Variable data = new Variable(Nd4j.create(new double[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {10, 11, 12},

        }), true);

        Average result = new Average(data);

        result.forwardWithInput();

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
