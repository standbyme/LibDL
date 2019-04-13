package LibDL.Tensor.Operator;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Tensor;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class GetTest {
    @Test
    public void testGet() {
        Constant data = new Constant(Nd4j.create(new double[][]{
                {0.3, 4.0, 2.9},
                {3.5, 2.2, 2.5},
                {0.5, 6, 6.5},
                {0.5, 6, 7.5},
        }), true);
        data.forward();

        Tensor result = data.get(0);

        result.forward();

        assertEquals(result.out, Nd4j.create(new double[]{0.3, 4.0, 2.9}));

        result.dout = Nd4j.create(new double[]{0.5, 1, 2});

        result.backward();
        assertEquals(data.dout, Nd4j.create(new double[][]{
                {0.5, 1, 2},
                {0, 0, 0},
                {0, 0, 0},
                {0, 0, 0}
        }));

        Tensor result2 = data.get(2);

        result2.forward();

        assertEquals(result2.out, Nd4j.create(new double[]{0.5, 6, 6.5}));

        result2.dout = Nd4j.create(new double[]{2, 1.0, 3});

        result2.backward();
        assertEquals(data.dout, Nd4j.create(new double[][]{
                {0, 0, 0},
                {0, 0, 0},
                {2, 1.0, 3},
                {0, 0, 0}
        }));

    }
}
