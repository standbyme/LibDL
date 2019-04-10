package vision.datasets;

import LibDL.Tensor.Constant;
import LibDL.nn.*;
import LibDL.optim.SGD;
import LibDL.utils.Pair;
import LibDL.utils.data.DataLoader;
import LibDL.utils.data.Dataset;
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
    public void testMNISTWithAutoEncoder() {

        MNIST mnist_train = new MNIST("resource/MNIST/", true);
        MNIST mnist_test = new MNIST("resource/MNIST/", false);
        Sequential encoder = new Sequential(
                new Dense(784, 100).withName("784-100"),
                new ReLU().withName("ReLU1"),
                new Dense(100, 10).withName("100-10"),
                new ReLU().withName("ReLU2"),
                new Dense(10, 1).withName("10-1"),
                new ReLU().withName("ReLU3")
        );

        Sequential decoder = new Sequential(
                new Dense(1, 10).withName("1-10"),
                new ReLU().withName("ReLU3"),
                new Dense(10, 100).withName("10-100"),
                new ReLU().withName("ReLU4"),
                new Dense(100, 784).withName("100-784"),
                new ReLU()
        );

        Sequential nn = new Sequential(
                encoder.withName("Encoder"),
                decoder.withName("Decoder")
        );

        for (Pair<INDArray, INDArray> e :
                new DataLoader(mnist_train, 10, false, false)) {
//            System.out.println(Arrays.toString(e.first.shape()));
//            System.out.println(Arrays.toString(e.second.shape()));
            nn.setInput(new Constant(e.first));

            MSELoss loss = new MSELoss(new Constant(e.first));
            loss.withName("Loss");
            loss.setInput(nn);
            LibDL.optim.SGD optim = new SGD(nn.parameters(), 0.0005f, 0.7f);


            for (int i = 0; i < 1; i++) {
                loss.forward();
                loss.backward();
                optim.step();
                System.out.println("time " + i + " " + loss.out.getRow(0));
            }

        }
    }

    @Test
    public void testMNISTWithLinear() {
        Dataset mnist_train = new MNIST("resource/MNIST/", true).reshapeData(784);
        Dataset mnist_test = new MNIST("resource/MNIST/", false);

        Sequential nn = new Sequential(
                new Dense(784, 100).withName("Dense784_100"),
                new ReLU().withName("RELU784_100"),
                new Dense(100, 10).withName("Dense100_10"),
                new ReLU().withName("RELU100_10"),
                new Dense(10, 10).withName("Dense10_10"),
                new Softmax().withName("SoftMax")
        );


        LibDL.optim.SGD optim = new SGD(nn.withName("NN").parameters(), 0.005f);

        for (Pair<INDArray, INDArray> e :
                new DataLoader(mnist_train, 100, false, false)) {

            CrossEntropyLoss loss = Functional.cross_entropy(
                    nn.predict(new Constant(e.first)),
                    new Constant(e.second));
            loss.backward();
            optim.step();

            System.out.println(loss.out.getRow(0));
        }

        nn.setInput(new Constant(mnist_test.data.reshape(10000, 784)));

        nn.forward();

        for (int i = 0; i < 100; i++) {
            System.out.println(nn.out.argMax(i) + ", " + mnist_test.target.getDouble(i));
        }

    }

}
