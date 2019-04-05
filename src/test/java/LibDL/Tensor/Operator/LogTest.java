package LibDL.Tensor.Operator;

import LibDL.Tensor.Constant;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class LogTest {
    @Test
    public void testLog() {
        Constant data_to_backward;
        Softmax softmax;
        INDArray target;
        data_to_backward = new Constant(Nd4j.create(new double[] { 1.0,  2.0,  3.0}), true);
        softmax = new Softmax(data_to_backward, 1);
        Log log = new Log(softmax);
        log.dout = Nd4j.create(new double[] {-2.81521177291870117188, -6.81521177291870117188, -6.81521177291870117188});
        log.forward();
        target = Nd4j.create(new double[] {-2.40760588645935058594, -1.40760588645935058594, -0.40760588645935058594});
        assertEquals(target, log.out);
        log.backward();
        target = Nd4j.create(new double[] {-1.33460175991058349609, -2.79049634933471679688,  4.12509870529174804688});
        assertEquals(target, data_to_backward.dout);
    }
}
