package LibDL.Tensor.Operator;

import LibDL.Tensor.Variable;
import LibDL.Tensor.Tensor;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
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

        data1.forwardWithInput();
        data2.forwardWithInput();

        Tensor result = new Concat(data1, data2);

        // forwardWithInput
        result.forwardWithInput();

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

    @Test
    public void testForwardInConv() {
        INDArray a = Nd4j.linspace(1, 6, 6).reshape(1, 2, 3);
        Variable input = new Variable(a, true);
        Concat concat = new Concat(input, 3, 1);
        concat.forwardWithInput();
        assertEquals(Nd4j.create(new double[][][] {{
                {1, 2, 3}, {4, 5, 6},{1, 2, 3}, {4, 5, 6},{1, 2, 3}, {4, 5, 6}
        }}), concat.out);
        concat.dout = Nd4j.concat(1, a, a, a);
        concat.backward();
        assertEquals(a, input.dout);
    }
}
