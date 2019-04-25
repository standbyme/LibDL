package LibDL.Tensor.Operator;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;


public class MultipleOutputTest {
    @Test
    public void test() {
        Variable v = new Variable(Nd4j.create(new double[]{7}), true);
        Tensor a = v.pow(3);
        Tensor b = a.mul(3).add(a.mul(5)).mul(2);
        // f(v) = ((v^3)*3+(v^3)*5)*2
        // df(v) = 48v^2dv
        b.grad = Nd4j.create(new double[]{1});
        b.backward();
        System.out.println(v.grad);
        assert v.grad.getDouble(0) == 2352.00;
    }
}
