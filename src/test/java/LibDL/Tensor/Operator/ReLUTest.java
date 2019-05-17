package LibDL.Tensor.Operator;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


import static org.junit.Assert.assertEquals;


public class ReLUTest {
    @Test
    public void testReLU() {
        Variable x1 = new Variable(Nd4j.randn(20, 30), true);

        Tensor u = new ReLU(x1);
        // forward
        INDArray gt = x1.data.gt(0).castTo(DataType.DOUBLE);
        INDArray out = x1.data.mul(gt);
        assertEquals(u.data, out);

        // backward
        u.sum().backward();
        assertEquals(x1.grad, gt);
    }
}
