package vision.datasets;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Operator.CrossEntropyLoss;
import LibDL.nn.*;
import LibDL.optim.SGD;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.Iterator;

public class MNISTTest {
    @Test
    public void testMNIST() {
        MNIST mnist_train = new MNIST("resource/MNIST/", true);
        MNIST mnist_test = new MNIST("resource/MNIST/", false);
//        System.out.println(Arrays.toString(mnist.data.shape()));
//        System.out.println(Arrays.toString(mnist_test.target.shape()));
        assert Arrays.equals(mnist_train.data.shape(), new long[]{60000, 28, 28});
        assert Arrays.equals(mnist_train.target.shape(), new long[]{60000, 1});

        assert Arrays.equals(mnist_test.data.shape(), new long[]{10000, 28, 28});
        assert Arrays.equals(mnist_test.target.shape(), new long[]{10000, 1});

        Iterator<INDArray[]> it_train = mnist_train.iterator();
        Iterator<INDArray[]> it_test = mnist_test.iterator();
//        System.out.println(Arrays.toString(it.next().getKey().shape()));
//        System.out.println(Arrays.toString(it.next().getValue().shape()));
        assert Arrays.equals(it_train.next()[0].shape(), (new long[]{28, 28}));
        assert Arrays.equals(it_train.next()[1].shape(), (new long[]{1, 1}));

        assert Arrays.equals(it_test.next()[0].shape(), (new long[]{28, 28}));
        assert Arrays.equals(it_test.next()[1].shape(), (new long[]{1, 1}));

        assert mnist_train.size() == 60000;
        assert mnist_test.size() == 10000;
    }

    private INDArray createOneHot(INDArray array, int sz) {
//        System.out.println(array.size(0));
        INDArray one_hot = Nd4j.zeros(array.size(0) * sz, 1);
        INDArray ones = Nd4j.onesLike(array);
        INDArray indices = Nd4j.linspace(0, array.size(0) - 1,
                array.size(0)).transpose();
//        System.out.println(Arrays.toString(array.shape()));
        System.out.println(Arrays.toString(one_hot.shape()));
        indices = indices.mul(sz).add(array);
        one_hot.put(new INDArrayIndex[]{NDArrayIndex.indices(indices.data().asLong()), NDArrayIndex.all()}, ones);
        return one_hot.reshape(array.size(0), sz);
//                    INDArray zeros = Nd4j.toFlattened(Nd4j.zerosLike(tensor.out));
//                    INDArray indices = Nd4j.linspace(0, argmax.size(0) - 1, argmax.size(0));
//                    indices = indices.mul(tensor.out.size(1)).add(argmax.transpose());
//                    zeros.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.indices(indices.data().asLong())}, dout);
//                    return zeros.reshape(tensor.out.shape());
    }

    @Test
    public void testMNISTWithLinear() {
        MNIST mnist_train = new MNIST("resource/MNIST/", true);
        MNIST mnist_test = new MNIST("resource/MNIST/", false);

        Sequential nn = new Sequential(
                new Linear(784, 100),
                new ReLU(),
                new Linear(100, 10),
                new ReLU(),
                new Linear(10, 10),
                new Softmax(1)
        );


        Constant data = new Constant(mnist_train.data.reshape(60000, 784));
        Constant target = new Constant(createOneHot(mnist_train.target, 10));

//        for (int i = 0; i < 10; i++)
//            System.out.println(target.value.getDouble(0, i));

        nn.setInput(data);

        CrossEntropy loss = new CrossEntropy(target);
        loss.setInput(nn);

        LibDL.optim.SGD optim = new SGD(nn.parameters(), 0.5f);


        for (int i = 0; i < 100; i++) {
            System.out.println("1 " + Arrays.toString(data.value.getRow(0).toDoubleVector()));
//            if(i!=0){
//
//            }
            loss.forward();
            System.out.println("2 " + Arrays.toString(data.value.getRow(0).toDoubleVector()));
            System.out.println("2 " + Arrays.toString(nn.out.getRow(0).toDoubleVector()));
            loss.backward();
            System.out.println("3 " + Arrays.toString(data.value.getRow(0).toDoubleVector()));
            System.out.println("3 " + Arrays.toString(nn.out.getRow(0).toDoubleVector()));
            optim.step();
            System.out.println("4 " + Arrays.toString(data.value.getRow(0).toDoubleVector()));
            System.out.println("4 " + Arrays.toString(nn.out.getRow(0).toDoubleVector()));
//            System.out.println(Arrays.toString(nn.out.getRow(0).toDoubleVector()));
//            System.out.println(Arrays.toString(nn.dout.getRow(0).toDoubleVector()));
            System.out.println(loss.out.getDouble(0));
            if (i % 50 == 0)
                System.out.println("time " + i);
        }

        nn.setInput(new Constant(mnist_test.data.reshape(10000, 784)));

        nn.forward();

        for (int i = 0; i < 100; i++) {
            System.out.println(nn.out.argMax(i) + ", " + mnist_test.target.getDouble(i));
        }

    }

}
