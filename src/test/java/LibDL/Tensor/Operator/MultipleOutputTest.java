package LibDL.Tensor.Operator;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;


public class MultipleOutputTest {
    @Test
    public void test0() {
        Variable v = new Variable(Nd4j.create(new double[]{7}), true);
        Tensor a = v.pow(3);
        Tensor b = a.mul(3).add(a.mul(5)).mul(2);
        // f(v) = ((v^3)*3+(v^3)*5)*2
        // df(v) = 48v^2dv
        b.backward();
        assert v.grad.getDouble(0) == 2352.00;
    }

    @Test
    public void test1_1() {
        Variable v = new Variable(Nd4j.create(new double[] {0.5}), true);
        Tensor x = v.mul(1);
        Tensor l = x.mul(5).add(x);

        l.backward();
        assert v.grad.getDouble(0) == 6.0;
    }

    @Test
    public void test1_2() {
        Variable v = new Variable(Nd4j.create(new double[] {0.5}), true);
        Tensor x = v.mul(1);
        Tensor l = x.add(x.mul(5));

        l.backward();
        assert v.grad.getDouble(0) == 6.0;
    }

    @Test
    public void test2() {
        Variable v = new Variable(Nd4j.create(new double[] {0.3}), true);
        Tensor x = v.mul(1);
        Tensor l = x.add(x);

        l.backward();
        assert v.grad.getDouble(0) == 2.0;
    }

    @Test
    public void test3() {
        Variable v = new Variable(Nd4j.create(new double[] {0.7}), true);
        Tensor x = v.mul(1);
        Tensor y = x.mul(2);
        Tensor z = x.mul(3);
        Tensor l = y.add(z);

        l.backward();
        assert v.grad.getDouble(0) == 5.0;
    }

}
