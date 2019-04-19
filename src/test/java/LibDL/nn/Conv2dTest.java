package LibDL.nn;

import LibDL.Tensor.Constant;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

public class Conv2dTest {
    @Test
    public void testConstruct() {
        Constant input = new Constant(Nd4j.linspace(1, 100, 100).reshape(2, 2, 5, 5));
        Conv2d conv2d = new Conv2d.Builder(6, 3, 3).groups(3).build();

        conv2d.W();

        INDArray a = Nd4j.rand(new int[] {3, 5, 6});
        a.broadcast(3, 10, 6);
    }
    @Test
    public void testMore() {
        INDArray a = Nd4j.rand(new int[] {3, 5, 4});
        INDArray b = Nd4j.rand(new int[] {1, 5, 1});
        System.out.println(b);
        b = b.broadcast(3, 5, 4);
        System.out.println(b);
        a.mul(b);
    }
    @Test
    public void testForward() {
        Constant input = new Constant(Nd4j.linspace(1, 100, 100).reshape(2, 2, 5, 5));
        Conv2d conv2d = new Conv2d.Builder(2, 2, 3, 3).padding(1, 1).groups(2).build();
        conv2d.setInput(input);
        conv2d.forward();
        System.out.println();
    }
}
