package LibDL.Tensor.Operator;

import LibDL.Tensor.Parameter;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class MaxTest {
    @Test
    public void testFxBWithSingleDim() {
        Parameter input = new Parameter(Nd4j.create(new double[][][][]{
                {
                        {{0.1, 1.0, 1.5}, {5.1, -1., 0.0}, {2.0, 3.0, 1.0}},
                        {{0.5, 0.6, 0.7}, {0.8, 0.9, 0.8}, {1.0, 1.1, 0.4}}
                },

                {
                        {{2.1, 5.1, 0.6}, {1.5, 0.2, 3.0}, {6.1, 2.1, 2.0}},
                        {{1.0, 1.0, 1.1}, {1.2, 1.3, 1.4}, {1.0, 0.2, 0.5}}
                }
        }));
        Max m = new Max(input, 1);
        INDArray expected = Nd4j.create(new double[][][]{
                {
                        {0.5, 1.0, 1.5}, {5.1, 0.9, 0.8}, {2.0, 3.0, 1.0}
                },

                {
                        {2.1, 5.1, 1.1}, {1.5, 1.3, 3.0}, {6.1, 2.1, 2.0}
                }
        });
        assertEquals(expected, m.data); // forward
        m.grad = Nd4j.onesLike(expected);
        m.backward();
        expected = Nd4j.create(new double[][][][]{
                {
                        {{0.0, 1.0, 1.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 1.0}},
                        {{1.0, 0.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, 0.0}}
                },

                {
                        {{1.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 1.0}},
                        {{0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}}
                }
        });
        assertEquals(expected, input.grad); // backward
    }
}
