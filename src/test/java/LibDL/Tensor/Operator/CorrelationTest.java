package LibDL.Tensor.Operator;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Parameter;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.junit.Assert.*;

public class CorrelationTest {

    @Test
    public void testCorrelation() {
        INDArray i = Nd4j.linspace(0, (64 * 50 * 576 - 1) / 1000, 64 * 50 * 576).reshape(64, 50, 576);
        INDArray w = Nd4j.linspace(0, 3.99, 400).reshape(1, 400, 1);
        Parameter input1 = new Parameter(i);
        Parameter weight1 = new Parameter(w);
        Concat concat = new Concat(input1, 8, 1);
        BroadcastMul broadcastMul = new BroadcastMul(concat, weight1, 2, 8, 1);
        Reshape reshape = new Reshape(broadcastMul,
                broadcastMul.data.shape()[0], 8,
                5 * 5 * 2, 24, 24);
        Sum sum = new Sum(reshape, 2);

        Parameter input2 = new Parameter(i);
        Parameter weight2 = new Parameter(w);
        Correlation m = new Correlation(input2, weight2, 24, 24, 2, 8, 1);

        m.grad = Nd4j.onesLike(m.data);
        sum.grad = Nd4j.onesLike(m.data);
        m.backward();
        sum.backward();

        assertEquals(Arrays.toString(sum.data.shape()), Arrays.toString(m.data.shape()));
        assertEquals(sum.data, m.data);
        assertEquals(input1.grad, input2.grad);
        assertEquals(weight1.grad, weight2.grad);

    }

}