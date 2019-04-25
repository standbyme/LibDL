package LibDL.example;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import LibDL.nn.*;
import LibDL.optim.RMSProp;
import LibDL.utils.Pair;
import LibDL.utils.data.DataLoader;
import LibDL.utils.data.Dataset;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import vision.datasets.MNIST;

import java.awt.*;
import java.util.Arrays;

public class MNISTExample {
//    @Test
//    public void testMNISTWithAutoEncoder() {
//
//        MNIST mnist_train = new MNIST("resource/MNIST/", true);
//        MNIST mnist_test = new MNIST("resource/MNIST/", false);
//        Sequential encoder = new Sequential(
//                new Dense(784, 100).withName("784-100"),
//                new ReLU().withName("ReLU1"),
//                new Dense(100, 10).withName("100-10"),
//                new ReLU().withName("ReLU2"),
//                new Dense(10, 1).withName("10-1"),
//                new ReLU().withName("ReLU3")
//        );
//
//        Sequential decoder = new Sequential(
//                new Dense(1, 10).withName("1-10"),
//                new ReLU().withName("ReLU3"),
//                new Dense(10, 100).withName("10-100"),
//                new ReLU().withName("ReLU4"),
//                new Dense(100, 784).withName("100-784"),
//                new ReLU()
//        );
//
//        Sequential nn = new Sequential(
//                encoder.withName("Encoder"),
//                decoder.withName("Decoder")
//        );
//
//        for (Pair<INDArray, INDArray> e :
//                new DataLoader(mnist_train, 10, false, false)) {
////            System.out.println(Arrays.toString(e.first.shape()));
////            System.out.println(Arrays.toString(e.second.shape()));
//            nn.setInput(new Variable(e.first));
//
//            MSELoss loss = new MSELoss(new Variable(e.first));
//            loss.withName("Loss");
//            loss.setInput(nn);
//            LibDL.optim.SGD optim = new SGD(nn.parameters(), 0.0005f, 0.7f);
//
//
//            for (int i = 0; i < 1; i++) {
//                loss.backward();
//                optim.step();
//                System.out.println("time " + i + " " + loss.out.getRow(0));
//            }
//
//        }
//    }

    public static void main(String[] args) {
        Dataset mnist_train = new MNIST("resource/MNIST/", true).reshapeData(784);
        Dataset mnist_test = new MNIST("resource/MNIST/", false);

        Sequential nn = new Sequential(
                new Dense(784, 200),
                new ReLU(),
                new Dense(200, 100),
                new ReLU(),
                new Dense(100, 20),
                new ReLU(),
                new Dense(20, 10)
//                new Softmax().withName("SoftMax")
        );
        System.out.println(nn);

        RMSProp optim = new RMSProp(nn.parameters(), 0.0002f, 0.99f, 5e-8);

        int cnt = 0;
        for (int epoch = 0; epoch < 10; epoch++) {
            for (Pair<INDArray, INDArray> batch :
                    new DataLoader(mnist_train, 500, false, false)) {
                optim.zero_grad();
                Tensor pred = nn.predict(new Variable(batch.first));
                Tensor target = new Variable(batch.second);
                Tensor loss = Functional.cross_entropy(pred, target);
                loss.backward();
                optim.step();
                if (cnt % 50 == 0) {
                    System.out.println("CNT: " + cnt + " " + loss.data.getRow(0));
                }
                cnt++;
            }
        }

        Tensor result = nn.predict(new Variable(mnist_test.reshapeData(784).data));

        INDArray out = MNIST.revertOneHot(result.data);

        int rightCnt = Arrays.stream(Transforms.abs(out.sub(mnist_test.target)).toDoubleVector())
                .filter(i -> i < 1e-6).toArray().length;
        System.out.println(rightCnt);
        assert (rightCnt > 9000);

    }
}
