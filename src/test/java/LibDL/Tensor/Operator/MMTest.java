package LibDL.Tensor.Operator;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;


public class MMTest {
    @Test
    public void main() {
        Variable x1 = new Variable(Nd4j.randn(2, 3), true);
        Variable y1 = new Variable(Nd4j.randn(3, 4), true);

        Tensor u = new MM(x1, y1);

        u.grad = Nd4j.onesLike(u.data);
        u.backward();

        Variable x2 = new Variable(x1.data.dup(), true);
        Variable y2 = new Variable(y1.data.dup(), true);

        Tensor v = new MM(x2, y2, true);
        v.grad = Nd4j.onesLike(v.data);
        v.backward();

        System.out.println(v.data);
        System.out.println(v.data.mul(2));
        assert u.data.equals(v.data);
        assert x1.grad.equals(x2.grad);
        assert y1.grad.equals(y2.grad);
    }
}
