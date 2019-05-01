package LibDL.Tensor.Operator;

import LibDL.Tensor.Constant;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

public class PaddingTest {
    @Test
    public void testOthers() {
        long[] shape = new long[5];
        long[] a = shape;
        System.out.println(shape);
        System.out.println(a);
    }
    @Test
    public void testForward() {
        Constant input = new Constant(Nd4j.rand(new int[] {2, 2, 3, 4}));
        int[] padding = new int[]{1, 0};
        int[] kernel = new int[]{2, 2};
        int[] stride = new int[]{3, 1};
        int[] dilation = new int[]{2, 1};
        Padding m = new Padding(input, padding, kernel, stride, dilation, true);
        System.out.println();
    }
}
