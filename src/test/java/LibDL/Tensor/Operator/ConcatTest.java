package LibDL.Tensor.Operator;

import LibDL.Tensor.Constant;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class ConcatTest {
    @Test
    public void testForward() {
        INDArray a = Nd4j.linspace(1, 6, 6).reshape(1, 2, 3);
        Constant input = new Constant(a, true);
        Concat concat = new Concat(input, 3, 1);
        concat.forward();
        assertEquals(Nd4j.create(new double[][][] {{
                {1, 2, 3}, {4, 5, 6},{1, 2, 3}, {4, 5, 6},{1, 2, 3}, {4, 5, 6}
        }}), concat.out);
        concat.dout = Nd4j.concat(1, a, a, a);
        concat.backward();
        assertEquals(a, input.dout);
    }
}
