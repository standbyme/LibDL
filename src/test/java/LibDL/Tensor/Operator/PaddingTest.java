package LibDL.Tensor.Operator;

import LibDL.Tensor.Constant;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;

public class PaddingTest {
    @Test
    public void testOthers() {
        INDArray a = Nd4j.rand(new int[] {3, 4, 2});
        System.out.println(a.shapeInfo());

        long[] b = new long[] {1, 2, 3};
        long[] c = new long[] {1, 2, 3, 4};
        b = Arrays.copyOf(c, 4);
        System.out.println(a);
        System.out.println();
        System.out.println(a.max(2));
        System.out.println(a.argMax(2));
        System.out.println();
        System.out.println(a.max(1));
        System.out.println(a.argMax(1));
        System.out.println();
        System.out.println(a.max(0));
        System.out.println(a.argMax(0));
    }
    @Test
    public void testForward() {
        Constant input = new Constant(Nd4j.rand(new int[] {2, 1, 3, 3}));
        int[] padding = new int[]{1, 1};
        int[] kernel = new int[]{2, 2};
        int[] stride = new int[]{3, 3};
        int[] dilation = new int[]{2, 2};
        Padding m = new Padding(input, kernel, padding, stride, dilation, true);
        System.out.println();
    }
}
