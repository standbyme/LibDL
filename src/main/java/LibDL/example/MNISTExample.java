package LibDL.example;

import LibDL.Tensor.Tensor;
import LibDL.nn.*;
import LibDL.Tensor.Constant;
import LibDL.optim.RMSProp;
import LibDL.utils.Pair;
import LibDL.utils.data.DataLoader;
import LibDL.utils.data.Dataset;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import vision.datasets.MNIST;

import java.util.Arrays;

public class MNISTExample {

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
                Tensor pred = nn.forward(new Constant(batch.first));
                Tensor target = new Constant(batch.second);
                Tensor loss = Functional.cross_entropy(pred, target);
                loss.backward();
                optim.step();
                if (cnt % 50 == 0) {
                    System.out.println("CNT: " + cnt + " " + loss.data.getRow(0));
                }
                cnt++;
            }
        }

        Tensor result = nn.forward(new Constant(mnist_test.reshapeData(784).data));

        INDArray out = MNIST.revertOneHot(result.data);

        int rightCnt = Arrays.stream(Transforms.abs(out.sub(mnist_test.target)).toDoubleVector())
                .filter(i -> i < 1e-6).toArray().length;
        System.out.println(rightCnt);
        assert (rightCnt > 9000);

    }
}
