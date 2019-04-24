package LibDL.Tensor.Operator;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;


public class AutogradTest {

    @Test
    public void testMulti1() {
        Tensor x = new Variable(Nd4j.create(new double[] {0.2}), true);
        Tensor y = x.mul(5);
        Tensor l = x.add(y);

        l.grad = Nd4j.create(new double[] {1.0});
        l.backward();

        assert l.data.equalsWithEps(Nd4j.create(new double[] {1.2}), 1e-3);
        assert x.grad.equalsWithEps(Nd4j.create(new double[] {6.}), 1e-3);
    }

    @Test
    public void testMulti2() {
        Tensor x = new Variable(Nd4j.create(new double[] {0.2}), true);
        Tensor y = x.mul(2);
        Tensor z = x.mul(3);
        Tensor l = y.add(z);

        l.grad = Nd4j.create(new double[] {1.0});
        l.backward();

        assert l.data.equalsWithEps(Nd4j.create(new double[] {1.0}), 1e-3);
        assert x.grad.equalsWithEps(Nd4j.create(new double[] {5.}), 1e-3);
    }

    @Test
    public void testMulti3() {
        Tensor x = new Variable(Nd4j.create(new double[] {0.5}), true);
        Tensor l = x.add(x);

        l.grad = Nd4j.create(new double[] {1.0});
        l.backward();

        assert l.data.equalsWithEps(Nd4j.create(new double[] {1.0}), 1e-3);
        assert x.grad.equalsWithEps(Nd4j.create(new double[] {2.}), 1e-3);
    }

    @Test
    public void testMulti4() {
        Tensor x = new Variable(Nd4j.create(new double[] {0.2}), true);
        Tensor x2 = x.mul(1);

        Tensor y = x2.mul(2);
        Tensor z = x2.mul(3);
        Tensor l = y.add(z);

        l.grad = Nd4j.create(new double[] {1.0});
        l.backward();

        assert l.data.equalsWithEps(Nd4j.create(new double[] {1.0}), 1e-3);
        assert x.grad.equalsWithEps(Nd4j.create(new double[] {5.}), 1e-3);
    }

}
