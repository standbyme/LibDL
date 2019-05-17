package LibDL.Tensor.Operator;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Parameter;
import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class BroadcastMulTest {
    @Test
    public void testBroadcastMul() {
        Variable input = new Variable(Nd4j.create(new double[][][]{
                {{1, 0}, {2, 1}, {1, 3}, {4, 2}, {1, 0}, {2, 1}, {1, 3}, {4, 2}},

                {{0, 1}, {1, 2}, {2, 3}, {5, 3}, {0, 1}, {1, 2}, {2, 3}, {5, 3}}
        }), true);
        Variable weight = new Variable(Nd4j.create(new double[][][]{{{2}, {3}, {0}, {0}, {0}, {0}, {4}, {5}}}), true);
        BroadcastMul broadcastMul = new BroadcastMul(input, weight, 2, 2 ,2);
        INDArray expected = Nd4j.create(new double[][][]{
                {{2, 0}, {6, 3}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {4, 12}, {20, 10}},

                {{0, 2}, {3, 6}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {8, 12}, {25, 15}}
        });
        assertEquals(expected, broadcastMul.data); // forward

        broadcastMul.grad = Nd4j.onesLike(input.data);
        broadcastMul.backward();
        assertEquals(Nd4j.create(new double[][][]{ // backward of input
                {{2, 2}, {3, 3}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {4, 4}, {5, 5}},

                {{2, 2}, {3, 3}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {4, 4}, {5, 5}}
        }), input.grad);
        assertEquals(Nd4j.create(new double[][][]{{{2}, {6}, {0}, {0}, {0}, {0}, {9}, {14}}}), weight.grad);
    }
}
