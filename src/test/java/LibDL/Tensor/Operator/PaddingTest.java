package LibDL.Tensor.Operator;

import LibDL.Tensor.Constant;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;

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
        assertEquals(a, m.data.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(1, 6), NDArrayIndex.interval(1, 7)));
        assert Arrays.equals(new long[]{1, 1, 7, 8}, m.data.shape());
    }
    @Test
    public void testForward() {
        Constant input = new Constant(Nd4j.rand(new int[] {2, 1, 3, 3}));
        int[] padding = new int[]{1, 1};
        int[] kernel = new int[]{2, 2};
        int[] stride = new int[]{3, 3};
        int[] dilation = new int[]{2, 2};
        Padding m = new Padding(input, kernel, padding, stride, dilation, true);
        assertEquals(input.data, m.data.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(1, 4), NDArrayIndex.interval(1, 4)));
        assert Arrays.equals(new long[]{2, 1, 6, 6}, m.data.shape());
    }
}
