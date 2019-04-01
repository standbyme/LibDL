package LibDL.Tensor.Operator;

import LibDL.Tensor.Constant;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class LogTest {
    @Test
    public void testLog() {

        Constant x = new Constant(Nd4j.create(new double[] {0.5, 1, 2.731, 10}), true);
        Log log = new Log(x);
        log.forward();
        assertEquals(log.out, Nd4j.create(new double[] {-0.693147181, 0, 1.00466784, 2.30258509}));

//        log.dout = Nd4j.create(new double[] {0.5, 1, 2, -2.5});
//        log.backward();
//        assertEquals(x.dout, Nd4j.create(new double[] {1.64872127, 2.71828183, 7.3890561, 0.0820849986}));
    }

}
