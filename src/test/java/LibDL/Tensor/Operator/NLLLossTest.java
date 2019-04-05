package LibDL.Tensor.Operator;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Tensor;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NLLLossTest {
    @Test
    public void testNLLLoss() {
        Constant input = new Constant(Nd4j.create(new double[][] {{-2.40, -1.42, -0.41}, {-2.00, -1.00, -0.00}}), true);
        Constant target = new Constant(Nd4j.create(new double[] {0, 1}).reshape(2));
        NLLLoss nll = new NLLLoss.Builder(input, target).reduction("mean").build();
        nll.forward();
        System.out.println(nll.out);
        nll.backward();
        System.out.println(input.dout);
    }
}
