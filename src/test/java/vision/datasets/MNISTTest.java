package vision.datasets;


import LibDL.utils.Pair;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

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
        assert Arrays.equals(mnist_train.target.shape(), new long[]{60000});

        assert Arrays.equals(mnist_test.data.shape(), new long[]{10000, 28, 28});
        assert Arrays.equals(mnist_test.target.shape(), new long[]{10000});

        Iterator<Pair<INDArray, INDArray>> it_train = mnist_train.iterator();
        Iterator<Pair<INDArray, INDArray>> it_test = mnist_test.iterator();
//        System.out.println(Arrays.toString(it_train.next().first.shape()));
//        System.out.println(Arrays.toString(it_train.next().second.shape()));
        assert Arrays.equals(it_train.next().first.shape(), (new long[]{1, 28, 28}));
        assert Arrays.equals(it_train.next().second.shape(), (new long[]{1}));

        assert Arrays.equals(it_test.next().first.shape(), (new long[]{1, 28, 28}));
        assert Arrays.equals(it_test.next().second.shape(), (new long[]{1}));

        assert mnist_train.size() == 60000;
        assert mnist_test.size() == 10000;
    }
}
