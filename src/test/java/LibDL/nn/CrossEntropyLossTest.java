package LibDL.nn;

import LibDL.Tensor.Constant;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

public class CrossEntropyLossTest {
    @Test
    public void testCEL() {
        Constant x = new Constant(Nd4j.create(new double[][] {{1, 2, 3}, {1, 2, 3}}), true);
        CrossEntropyLoss cel = new CrossEntropyLoss(new Constant(Nd4j.create(new double[] {1, 0}).reshape(2)));
        cel.setInput(x);
        cel.forward();
        System.out.println(cel.out);
        cel.backward();
        System.out.println(x.dout);
    }
}
