package LibDL.Tensor.Operator;

import LibDL.Tensor.Constant;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

public class main {
    @Test
    public void testSoftmax() {
        Constant data = new Constant(Nd4j.create(new double[]{0.3, 2.9, 4.0}));


        Softmax result = new Softmax(data);

        result.forward();

        double a = result.out.getDouble(0);
        double b = result.out.getDouble(1);
        double c = result.out.getDouble(2);

        assert a==0.018211273476481438;
        assert b==0.24519184231758118;
        assert c==0.736596941947937;
    }
}
