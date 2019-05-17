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
        INDArray a = Nd4j.rand(new int[] {1, 1, 5, 6});
        Constant input = new Constant(a);

        int[] padding = new int[]{1, 1};
        int[] kernel = new int[]{2, 2};
        int[] stride = new int[]{3, 3};
        int[] dilation = new int[]{1, 1};

        Padding m = new Padding(input, kernel, padding, stride, dilation, true);
        System.out.println(m.data);
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
