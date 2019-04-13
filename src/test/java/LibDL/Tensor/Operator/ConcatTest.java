package LibDL.Tensor.Operator;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Tensor;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class ConcatTest {
    @Test
    public void testConcat() {
        Constant data1 = new Constant(Nd4j.create(new double[][]{
                {1, 2, 3},
                {2, 3, 4},
        }), true);
        Constant data2 = new Constant(Nd4j.create(new double[][]{
                {3, 4, 6},
                {6, 7, 8},
        }), true);

        data1.forward();
        data2.forward();

        Tensor result = new Concat(data1, data2);

        // forward
        result.forward();

        assertEquals(result.out, Nd4j.create(new double[][]{
                {1, 2, 3},
                {2, 3, 4},
                {3, 4, 6},
                {6, 7, 8},
        }));

        // backward
        result.dout = Nd4j.create(new double[][]{
                {1, 2, 2},
                {2, 5, 1},
                {3, 6, 3},
                {5, 3, 4},
        });

        result.backward();

        assertEquals(data1.dout, Nd4j.create(new double[][]{
                {1, 2, 2},
                {2, 5, 1},
        }));
        assertEquals(data2.dout, Nd4j.create(new double[][]{
                {3, 6, 3},
                {5, 3, 4},
        }));

    }
}
