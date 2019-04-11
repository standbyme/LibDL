package LibDL.Tensor.Operator;

import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class NLLLossTest {
    @Test
    public void testNLLLoss() {
        Variable input = new Variable(Nd4j.create(new double[][]{{-2.40, -1.42, -0.41}, {-2.00, -1.00, -0.00}}), true);
        Variable target = new Variable(Nd4j.create(new double[]{0, 1}).reshape(2));
        NLLLoss nll = new NLLLoss.Builder(input, target).reduction("mean").build();
        nll.forward();
        assertEquals(Nd4j.create(new double[] {1.7}).reshape(1), nll.out);
        nll.backward();
        assertEquals(Nd4j.create(new double[][] {{-0.5,  0.0,  0.0}, { 0.0, -0.5,  0.0}}), input.dout);
    }
}
