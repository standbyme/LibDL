package LibDL.Tensor.Operator;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class BroadcastAddTest {

    @Test
    public void addVectorTest() {
        Variable X = new Variable(Nd4j.randn(new int[]{2, 3, 4}), true);
        Variable Y = new Variable(Nd4j.ones(1, 4), true);

        Tensor Z = X.addVector(Y);

        assertEquals(Z.data, X.add(1).data);
    }

    @Test
    public void testForConv2d() {
        Variable input = new Variable(Nd4j.rand(new int[]{2, 4, 3, 3}), true);
        Variable B = new Variable(Nd4j.rand(new int[]{4}), true);
        AddVector addVector = new AddVector(input, B, true);
        addVector.grad = Nd4j.linspace(0, 71, 72).reshape(2, 4, 3, 3);
        addVector.backward();
        assertEquals(Nd4j.linspace(0, 71, 72).reshape(2, 4, 3, 3), input.grad);
        assertEquals(Nd4j.create(new double[]{396, 558, 720, 882}), B.grad);
    }
}
