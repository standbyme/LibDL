package LibDL.Tensor.Operator;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Parameter;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.junit.Assert.*;

public class CorrelationTest {

    @Test
    public void testCorrelation() {
//        INDArray i = Nd4j.linspace(0, (64 * 50 * 576 - 1) / 1000, 64 * 50 * 576).reshape(64, 50, 576);
//        INDArray w = Nd4j.linspace(0, 3.99, 400).reshape(1, 400, 1);
        int N = 64;
        int ic = 6;
        int oc = 9;
        int groups = 3;
        long ah = 24;
        long aw = 25;
        long fh = 5;
        long fw = 6;
        long size = fh * fw;
        INDArray i = Nd4j.rand(new int[]{N, (int) size * ic, (int) ah * (int) aw});
        INDArray w = Nd4j.rand(new int[]{1, (int) size * ic * oc, 1});
        Parameter input1 = new Parameter(i);
        Parameter weight1 = new Parameter(w);
        Concat concat = new Concat(input1, oc, 1);
        BroadcastMul broadcastMul = new BroadcastMul(concat, weight1, ic, oc, groups);
        Reshape reshape = new Reshape(broadcastMul,
                broadcastMul.data.shape()[0], oc,
                fh * fw * ic, ah, aw);
        Sum sum = new Sum(reshape, 2);

        Parameter input2 = new Parameter(i);
        Parameter weight2 = new Parameter(w);
        Correlation m = new Correlation(input2, weight2, ah, aw, ic, oc, groups);

        sum.grad = Nd4j.onesLike(m.data);
        m.grad = Nd4j.onesLike(m.data);
        sum.backward();
        m.backward();

        assertEquals(Arrays.toString(sum.data.shape()), Arrays.toString(m.data.shape()));
        assertEquals(sum.data, m.data);
        assertEquals(input1.grad, input2.grad);
        assertEquals(weight1.grad, weight2.grad);

    }

}