package LibDL.Tensor.Operator;

import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

public class BroadcastMulTest {
    @Test
    public void testBackward() {
        Variable input = new Variable(Nd4j.rand(new int[] {2, 48, 3}), true);
        Variable weight = new Variable(Nd4j.rand(new int[] {1, 48, 1}), true);
        BroadcastMul broadcastMul = new BroadcastMul(input, weight, 4, 4 ,2);
        broadcastMul.grad = Nd4j.rand(new int[] {2, 48, 3});
        broadcastMul.backward();
    }
}
