package LibDL.nn;

import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class SoftmaxCrossEntropyLossTest {
    @Test
    public void testCEL() {
        Variable x = new Variable(Nd4j.create(new double[][]{{1, 2, 3}, {1, 2, 3}, {1, 2, 3}}), true);
        SoftmaxCrossEntropyLoss cel = new SoftmaxCrossEntropyLoss(new Variable(Nd4j.create(new double[]{1, 0, 2}).reshape(3)));
        cel.setInput(x);
        assertEquals(Nd4j.create(new double[] {1.40760576725006103516}).reshape(1), cel.out);
        cel.backward();
        assertEquals(Nd4j.create(new double[][] {
                { 0.03001019358634948730, -0.25175717473030090332,  0.22174701094627380371},
                {-0.30332314968109130859,  0.08157616853713989258,  0.22174701094627380371},
                { 0.03001019358634948730,  0.08157616853713989258, -0.11158633232116699219}
        }), x.dout);
    }
}
