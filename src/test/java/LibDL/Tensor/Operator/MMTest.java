package LibDL.Tensor.Operator;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;


public class MMTest {
    @Test
    public void testMM() {
        Variable x1 = new Variable(Nd4j.randn(20, 30), true);
        Variable y1 = new Variable(Nd4j.randn(30, 40), true);

        Tensor u = new MM(x1, y1);

        u.grad = Nd4j.onesLike(u.data);
        u.backward();

        Variable x2 = new Variable(x1.data.dup(), true);
        Variable y2 = new Variable(y1.data.dup(), true);

        Tensor v = new MM(x2.reshape(2, 10, 30), y2, true);
        v.grad = Nd4j.onesLike(v.data);
        v.backward();

        // forward
        assert Arrays.equals(v.data.shape(), new long[]{2, 10, 40});
        assert u.data.equals(v.data.reshape(20, 40));
        // backward
        assert x1.grad.equals(x2.grad);
        assert y1.grad.equals(y2.grad);
    }
}
