package LibDL.nn;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

public class FunctionalTest {
    @Test
    public void sigmoidTest() {
        Tensor input = new Variable(Nd4j.linspace(-10, 10, 10), true);
        Tensor sigmoid = (Functional.sigmoid(input)).sum();
        System.out.println(sigmoid);
        assert sigmoid.data.getDouble(0) == 5.0;

//        sigmoid.grad=Nd4j.onesLike(input.data);
        sigmoid.backward();
        System.out.println(input.grad);

        assert input.grad.equalsWithEps(Nd4j.create(new double[]
                {4.5396e-5, 0.0004, 0.0038, 0.0333, 0.1863, 0.1863, 0.0333, 0.0038, 0.0004, 4.5396e-5}
        ), 0.01);
    }
}
