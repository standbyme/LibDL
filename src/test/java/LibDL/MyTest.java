package LibDL;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastCopyOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastEqualTo;
import org.nd4j.linalg.api.ops.impl.shape.Broadcast;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.FModOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.FloorModOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.OldFModOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.Assert.assertEquals;

public class MyTest {

    @Test
    public void testMod() {
        INDArray a = Nd4j.create(new double[][]{{1, 2.1, 13}});//.reshape(3, 1);
        INDArray b = Nd4j.create(new double[][]{{1, 5, 2.1}});;
        INDArray c = a.dup();
        Nd4j.getExecutioner().exec(new OldFModOp(a, b, c));
        assertEquals(Nd4j.create(new double[][]{{0, 2.1, 0.4}}), c);

        a = Nd4j.create(new double[][]{{-1, -2.1, -13}});//.reshape(3, 1);
        b = Nd4j.create(new double[][]{{ 1, 5, 2}});;

        Nd4j.getExecutioner().execAndReturn(new OldFModOp(a, b, c));
        assertEquals(Nd4j.create(new double[][]{{0, -2.1, -1}}), c);
    }

    @Test
    public void testIntDiv() {
        INDArray a = Nd4j.create(new double[][]{{1, 2.1, 3}});//.reshape(3, 1);

//        Nd4j.setDataType(DataBuffer.Type.INT);
        INDArray b = Nd4j.create(new double[][]{{1}});

        INDArray c = a.dup();
        Nd4j.getExecutioner().execAndReturn(new OldFModOp(a, b, c));
        System.out.println(c);
    }
    @Test
    public void testBroadcast() {
        INDArray a = Nd4j.create(new double[][]{{1, 0, 13}, {1, 0, 13}});//.reshape(3, 1);
        INDArray b = Nd4j.create(new double[][]{{1, 0, 13}}).reshape(3);
        INDArray c = Nd4j.create(new double[][]{{0, 0, 0},{0, 0, 0}});
        Nd4j.getExecutioner().exec(new BroadcastCopyOp(a, b,c , 1));
        a = Nd4j.rand(new int[]{3,2, 3, 2, 3});
        b = Nd4j.rand(new int[]{3,2, 2, 3});
        c = Nd4j.rand(new int[]{3, 2, 3, 2, 3, 3});
        Nd4j.getExecutioner().exec(new BroadcastCopyOp(a, b,c.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0)) , 0, 1, 3, 4));
    }

    @Test
    public void testGet() {
        INDArray a = Nd4j.create(new double[][]{{1, 0, 13}, {1, 0, 13}});
        INDArray b = a.get(NDArrayIndex.point(0), NDArrayIndex.point(0));
        a.putScalar(new int[]{0,0}, 2);
    }

    @Test
    public void testInterval() {
        INDArray x = Nd4j.randn(10, 8);
        System.out.println(x.get(NDArrayIndex.all(), NDArrayIndex.interval(4, 3, 9)));
        System.out.println(x.get(NDArrayIndex.all(), NDArrayIndex.interval(4, 3, 8)));
    }

    @Test
    public void testReshape() {
        INDArray a, b, c, d;
        a = Nd4j.rand(10, 6);
        c = a.dup();
        b = a.reshape(1, -1);
        d = Nd4j.toFlattened(c);
        b.putScalar(0, 0, 5);
        d.putScalar(0, 0, 5);

    }
}
