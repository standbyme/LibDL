package LibDL.nn;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;


public class MultipleOutputTest {
    @Test
    public void test() {
        Variable a = new Variable(Nd4j.create(new double[]{7}), true);
        Tensor b = a.mul(3).add(a.mul(5)).mul(2);
        b.grad = Nd4j.create(new double[]{1});
        b.backward();
        System.out.println(a.grad);
    }
}
