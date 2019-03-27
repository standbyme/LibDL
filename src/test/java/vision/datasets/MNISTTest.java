package vision.datasets;

import LibDL.Tensor.Constant;
import LibDL.nn.*;
import LibDL.optim.SGD;
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

    @Test
    public void testMNISTWithLinear() {
        MNIST mnist_train = new MNIST("resource/MNIST/", true);
        MNIST mnist_test = new MNIST("resource/MNIST/", false);

        Constant data = new Constant(mnist_train.data.reshape(60000, 784));
        Constant target = new Constant(mnist_train.target);

        Sequential nn = new Sequential(
                new Linear(784, 100),
                new ReLU(),
                new Linear(100, 10),
                new ReLU(),
                new Linear(10, 1),
                new ReLU()
        );

        nn.setInput(data);

        MSELoss loss = new MSELoss(target);
        loss.setInput(nn);

        LibDL.optim.SGD optim = new SGD(nn.parameters(), 0.0005f);


        for (int i = 0; i < 100; i++) {
            loss.forward();
            loss.backward();
            optim.step();
            System.out.println(loss.out.getDouble(0));
            if (i % 50 == 0)
                System.out.println("time " + i);
        }
        nn.setInput(new Constant(mnist_test.data.reshape(10000, 784)));

        nn.forward();

        for (int i = 0; i < 100; i++) {
            System.out.println(nn.out.getDouble(i) + ", " + mnist_test.target.getDouble(i));
        }

    }

}
