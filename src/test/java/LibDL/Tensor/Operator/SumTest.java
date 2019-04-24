package LibDL.Tensor.Operator;

import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

public class SumTest {
    @Test
    public void testForward() {
        Variable input = new Variable(Nd4j.linspace(1, 192, 192).reshape(2, 2, 8, 6));
        System.out.println(input.value);
        Sum sum = new Sum(input, 0,3);
        sum.forwardWithInput();
        System.out.println(sum.out);
    }
}
