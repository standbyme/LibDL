package LibDL.Tensor.Operator;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
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


        Tensor result = new Concat(data1, data2);

        assertEquals(result.data, Nd4j.create(new double[][]{
                {1, 2, 3},
                {2, 3, 4},
                {3, 4, 6},
                {6, 7, 8},
        }));

        // backward
        result.grad = Nd4j.create(new double[][]{
                {1, 2, 2},
                {2, 5, 1},
                {3, 6, 3},
                {5, 3, 4},
        });

        result.backward();

        assertEquals(data1.grad, Nd4j.create(new double[][]{
                {1, 2, 2},
                {2, 5, 1},
        }));
        assertEquals(data2.grad, Nd4j.create(new double[][]{
                {3, 6, 3},
                {5, 3, 4},
        }));

    }

    @Test
    public void testForwardInConv() {
        INDArray a = Nd4j.linspace(1, 6, 6).reshape(1, 2, 3);
        Variable input = new Variable(a, true);
        Concat concat = new Concat(input, 3, 1);
        assertEquals(Nd4j.create(new double[][][]{{
                {1, 2, 3}, {4, 5, 6}, {1, 2, 3}, {4, 5, 6}, {1, 2, 3}, {4, 5, 6}
        }}), concat.data);
        concat.grad = Nd4j.concat(1, a, a, a);
        concat.backward();
        assertEquals(Nd4j.create(new double[][][]{
                {
                        {3, 6, 9}, {12, 15, 18}
                }
        }), input.grad);
    }
}
