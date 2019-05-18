package LibDL.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastCopyOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class BenchmarkExample {

    public static void main(String[] args) throws InterruptedException {
//        INDArray a = Nd4j.rand(new int[]{64, 25 * 50, 24 * 24});
//        INDArray b = Nd4j.rand(new int[]{ 1, 25 * 50, 1});
//        INDArray c = Nd4j.zerosLike(a);
//        a = Nd4j.rand(new int[]{2, 4, 3});
//        b = Nd4j.rand(new int[]{ 1, 4, 1});
//        test();
//        test1(a, b);
//        test2(a, b);
        Thread.sleep(4000);
        for (int i = 1; i <= 1000000; i++) {
//            compareBroadcast();
            compToFlattened();
            System.out.println(i);
        }
        System.exit(0);

//        for (int i = 0; i < 20; i++) {
//            f5(a, b);
//        }
//        for (int i = 0; i < 20; i++) {
//            f1(a, b);
//        }
//        for (int i = 0; i < 20; i++) {
//            f2(a, b);
//        }
//        for (int i = 0; i < 20; i++) {
//            f3(a, b);
//        }
//        for (int i = 0; i < 20; i++) {
//            f4(a, b);
//        }

//        test(a, b);
    }
    private static void f1(INDArray a, INDArray b) {
        a.mul(b.broadcast(a.shape()));
    }
    private static void f2(INDArray a, INDArray b) {
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 24 * 24; j++) {
                a.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)).reshape(1, 200, 1).muli(b);
            }
        }
    }
    private static void f3(INDArray a, INDArray b) {
        for (int i = 0; i < 64; i++) {

            a.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()).reshape(1, 200, 24 * 24).muli(b.broadcast(1, 200, 24 * 24));

        }
    }
    private static void f4(INDArray a, INDArray b) {
        b.broadcast(64, 25 * 8, 24 * 24);
    }
    private static void f5(INDArray a, INDArray b) {
        Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(a, b, a, 1));
    }
    private static void test1(INDArray a, INDArray b) {
        INDArray d, c = Nd4j.onesLike(a);
        Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(a, b, c, 1));
        a.muli(2);
        System.out.println("test1:\n" + a);
    }
    private static void test2(INDArray a, INDArray b) {
        INDArray c = Nd4j.onesLike(a);
        Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(a, b, c, 1));
        assert a.muli(b.broadcast(a.shape())).equals(c);
    }
    private static void test() {
        INDArray a = Nd4j.create(new double[]{1, 2, 3});
        INDArray b;
        System.out.println(a);
        a.muli(2);
        System.out.println(a);
        b = a.muli(2);
        System.out.println(a);
    }
    private static void f6(INDArray a, INDArray b) {
        INDArray c = b.broadcast(a.shape());
    }

    private static void f7(INDArray a, INDArray b) {
        INDArray c = Nd4j.zerosLike(a);
        Nd4j.getExecutioner().exec(new BroadcastCopyOp(a, b ,c, 1));
    }
    private static void compareBroadcast() {
        INDArray A, B, C, a, b, c;
        A = Nd4j.rand(new int[]{64, 16, 200, 24, 24});
        B = Nd4j.rand(new int[]{64, 16, 1, 24, 24});
        C = Nd4j.zerosLike(A);

        b = B.dup();

        a = A.dup();
        c = C.dup();
        commonBroadcast(a, b, c);

        a = A.dup();
        c = C.dup();
        broadcastCopy11(a, b, c);

        a = A.dup();
        c = C.dup();
        broadcastCopy12(a, b, c);

        a = A.dup();
        c = C.dup();
        broadcastCopy21(a, b, c);

        a = A.dup();
        c = C.dup();
        broadcastCopy22(a, b, c);
    }
    private static void broadcastCopy11(INDArray a, INDArray b, INDArray c) {
        Nd4j.getExecutioner().exec(new BroadcastCopyOp(a, b ,c, 0, 1, 3, 4));
    }
    private static void broadcastCopy12(INDArray a, INDArray b, INDArray c) {
        Nd4j.getExecutioner().exec(new BroadcastCopyOp(a, b ,a, 0, 1, 3, 4));
    }
    private static void broadcastCopy21(INDArray a, INDArray b, INDArray c) {
        Nd4j.getExecutioner().execAndReturn(new BroadcastCopyOp(a, b ,c, 0, 1, 3, 4));
    }
    private static void broadcastCopy22(INDArray a, INDArray b, INDArray c) {
        Nd4j.getExecutioner().execAndReturn(new BroadcastCopyOp(a, b ,a, 0, 1, 3, 4));
    }
    private static void commonBroadcast(INDArray a, INDArray b, INDArray c) {
        c = b.broadcast(a.shape());
    }

    private static void compareExecAndReturn() {
        INDArray a, b, c;

        a = Nd4j.rand(new int[]{64, 25 * 50, 24 * 24});
        b = Nd4j.rand(new int[]{ 1, 25 * 50, 1});
        c = Nd4j.zerosLike(a);
        execAndReturn2(a, b, c);

//        a = Nd4j.rand(new int[]{64, 25 * 50, 24 * 24});
//        b = Nd4j.rand(new int[]{ 1, 25 * 50, 1});
//        c = Nd4j.zerosLike(a);
//        execAndReturn1(a, b, c);

//        a = Nd4j.rand(new int[]{64, 25 * 50, 24 * 24});
//        b = Nd4j.rand(new int[]{ 1, 25 * 50, 1});
//        c = Nd4j.zerosLike(a);
//        exec1(a, b, c);

        a = Nd4j.rand(new int[]{64, 25 * 50, 24 * 24});
        b = Nd4j.rand(new int[]{ 1, 25 * 50, 1});
        c = Nd4j.zerosLike(a);
        execAndReturn(a, b, c);

//        a = Nd4j.rand(new int[]{64, 25 * 50, 24 * 24});
//        b = Nd4j.rand(new int[]{ 1, 25 * 50, 1});
//        c = Nd4j.zerosLike(a);
//        exec(a, b, c);
    }
    private static void exec(INDArray a, INDArray b, INDArray c) {
        Nd4j.getExecutioner().exec(new BroadcastMulOp(a, b, c, 1));
    }
    private static void exec1(INDArray a, INDArray b, INDArray c) {
        Nd4j.getExecutioner().exec(new BroadcastMulOp(a, b, a, 1));
    }
    private static void execAndReturn(INDArray a, INDArray b, INDArray c) {
        Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(a, b, c, 1));
    }
    private static void execAndReturn1(INDArray a, INDArray b, INDArray c) {
        Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(a, b, a, 1));
    }
    private static void execAndReturn2(INDArray a, INDArray b, INDArray c) {

    }

    private static void compToFlattened() {
        INDArray A = Nd4j.rand(1024, 16);
        INDArray a;

        a = A.dup();
        reshape(a);

        a = A.dup();
        flattened(a);
    }
    private static void reshape(INDArray a) {
        INDArray b = a.reshape(-1);
    }
    private static void flattened(INDArray a) {
        INDArray b = Nd4j.toFlattened(a);
    }
}
