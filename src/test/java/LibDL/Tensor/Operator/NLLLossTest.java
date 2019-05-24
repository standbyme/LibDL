package LibDL.Tensor.Operator;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class NLLLossTest {
    @Test
    public void testNLLLoss1() {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        Variable input = new Variable(Nd4j.create(new double[][]{{-2.40, -1.42, -0.41}, {-2.00, -1.00, -0.00}}), true);
        Constant target = new Constant(Nd4j.create(new double[]{0, 1}).reshape(2));
        NLLLoss nll = new NLLLoss.Builder(input, target).reduction("mean").build();
        assertEquals(Nd4j.create(new double[]{1.7}), nll.data.reshape(1));
        nll.backward();
        assertEquals(Nd4j.create(new double[][]{{-0.5, 0.0, 0.0}, {0.0, -0.5, 0.0}}), input.grad);
    }

    @Test
    public void testNLLLoss2() {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        Variable input = new Variable(Nd4j.create(new double[][]{{-2.40, -1.42, -0.41, -2.00, -1.00, -0.00}}), true);
        Constant target = new Constant(Nd4j.create(new double[]{1}).reshape(1));
        NLLLoss nll = new NLLLoss.Builder(input, target).reduction("mean").build();
        assertEquals(Nd4j.create(new double[]{1.42}), nll.data.reshape(1));
        nll.backward();
        assertEquals(Nd4j.create(new double[][]{{0.0, -1.0, 0.0, 0.0, 0.0, 0.0}}), input.grad);
    }
}
