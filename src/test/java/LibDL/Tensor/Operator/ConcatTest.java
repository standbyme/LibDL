package LibDL.Tensor.Operator;

import LibDL.Tensor.Variable;
import LibDL.Tensor.Tensor;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class ConcatTest {
    @Test
    public void testConcat() {
        Variable data1 = new Variable(Nd4j.create(new double[][]{
                {1, 2, 3},
                {2, 3, 4},
        }), true);
        Variable data2 = new Variable(Nd4j.create(new double[][]{
                {3, 4, 6},
                {6, 7, 8},
        }), true);


        Tensor result = new Concat(data1, data2);

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
