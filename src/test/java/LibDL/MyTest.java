package LibDL;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
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
import org.nd4j.linalg.indexing.INDArrayIndex;
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

    @Test
    public void testFloatPointer() {
        INDArray a = Nd4j.rand(20, 2);
        INDArray b = a.dup();
        DataBuffer dataBuffer = a.data();
        FloatPointer floatPointer = (FloatPointer) dataBuffer.addressPointer();

        floatPointer.position(2).put(new float[]{1, 2, 3, 4}, 2, 5);
        floatPointer.position(2).put(new float[]{1, 2, 3, 4}, 2, 5);
        System.out.println(floatPointer.position());

        float[] f = new float[]{2, 3};
        floatPointer.get(f, 0, 2);

        INDArray d = Nd4j.create(new double[]{2.0, 3.0, 1.0, 4.0});

        Pointer df = d.get(NDArrayIndex.all(), NDArrayIndex.indices(0, 2)).data().pointer();

        assertEquals(a, b);
    }

    @Test
    public void testFloatPointer_Unfold1() {
        INDArray x = Nd4j.rand(new int[]{64, 25, 576});
        INDArray y = Nd4j.zerosLike(x);

        for (int i = 0; i < 576; i++) {
            INDArrayIndex[] indArrayIndices = new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i)};
            y.put(indArrayIndices, x.get(indArrayIndices));
        }

        assertEquals(x, y);
    }

    @Test
    public void testFloatPointer_Unfold2() {
        INDArray x = Nd4j.rand(new int[]{64, 25, 576});
        INDArray y = Nd4j.zerosLike(x);

        float[] f = new float[64 * 25];

        FloatPointer floatPointerX = (FloatPointer) x.data().pointer();

        FloatPointer floatPointerY = (FloatPointer) y.data().pointer();

        int len = f.length;
        for (int i = 0; i < 576; i++) {

//            floatPointerX.position(i * len).get(f, 0, len);
//            floatPointerY.position(i * len).put(f, 0, len);

            floatPointerY.position(i * len).put(floatPointerX.position(i * len));

        }

        floatPointerX.position(0);
        floatPointerY.position(0);

        assertEquals(x, y);
    }
}
