package LibDL.example;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Tensor;
import LibDL.nn.*;
import LibDL.optim.RMSProp;
import LibDL.optim.SGD;
import LibDL.utils.Pair;
import LibDL.utils.data.DataLoader;
import LibDL.utils.data.Dataset;
import org.nd4j.linalg.api.ndarray.INDArray;
import vision.datasets.MNIST;

public class CNNExample {

    public static void main(String[] args) {
        Dataset mnist_train = new MNIST("resource/MNIST/", true).reshapeData(1, 28, 28);
        Dataset mnist_test = new MNIST("resource/MNIST/", false).reshapeData(1, 28, 28);

        Sequential nn = new Sequential(
                new Conv2d.Builder(1, 8, 5).build(),
                new ReLU(),
                new MaxPool2d.Builder(2).build(),
                new Conv2d.Builder(8, 16, 5).build(),
                new ReLU(),
                new MaxPool2d.Builder(2).build(),
                new Reshape(60, 4 * 4 * 16),
                new Dense(4 * 4 * 16, 100),
                new ReLU(),
                new Dense(100, 10)
        );
        System.out.println(nn);

        SGD optimizer = new SGD(nn.parameters(), 0.01f, 0.50f);

        DataLoader dataLoader = new DataLoader(mnist_train, 60, false, false);
        int cnt = 0;
        for (Pair batch : dataLoader) {
            optimizer.zero_grad();
            Tensor predict = nn.forward(new Constant(((INDArray) batch.first).div(255)));
            Tensor target = new Constant((INDArray) batch.second);
            Tensor loss = Functional.cross_entropy(predict, target);
            loss.backward();
            optimizer.step();
            cnt++;
            if (cnt % 10 == 0) {
                System.out.println("batch: " + cnt + " " + loss.data.getRow(0));
            }
        }

        dataLoader = new DataLoader(mnist_test, 60, false, false);
        double l = 0.0;
        cnt = 0;
        for (Pair batch : dataLoader) {
            Tensor predict = nn.forward(new Constant(((INDArray) batch.first).div(255)));
            Tensor target = new Constant((INDArray) batch.second);
            Tensor loss = Functional.cross_entropy(predict, target);
            l += loss.data.sumNumber().doubleValue();
            cnt++;
            System.out.println(cnt + "\taverage loss: " + l /  cnt);
        }
        System.out.println("\n\naverage loss: " + l /  cnt);
    }

}
