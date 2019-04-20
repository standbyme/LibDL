package LibDL.example;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import LibDL.nn.*;
import LibDL.optim.Parameters;
import LibDL.optim.RMSProp;
import LibDL.optim.SGD;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.stream.IntStream;

public class XORExample {

    public static void main(String[] args) {
        testXOR();
        testXORWithMultipleInput();
    }

    public static void testXOR() {
        Variable data = new Variable(Nd4j.create(new double[][]{{1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.0, 0.0}}));
        Variable target = new Variable(Nd4j.create(new double[][]{{1.0}, {0.0}, {1.0}, {0.0}}));

        Sequential nn = new Sequential(new Dense(2, 5), new ReLU(), new Dense(5, 1));
        nn.setInput(data);

        MSELoss loss = new MSELoss(target);
        loss.setInput(nn);

        SGD optimizer = new SGD(new Parameters(nn), 0.1f);

        for (int epoch = 1; epoch <= 1000; epoch++) {
            loss.forwardWithInput();
            loss.backward();
            optimizer.step();
        }

        IntStream.rangeClosed(0, 3).forEach(i -> {
            assert Math.abs(target.value.getInt(i) - nn.out.getDouble(i)) < 0.1;
        });
    }

    public static void testXORWithMultipleInput() {
        Variable data = new Variable(
                Nd4j.create(new double[][]{{1.0, 0.0},
                        {1.0, 1.0},
                        {0.0, 1.0},
                        {0.0, 0.0}}));

        Variable target = new Variable(
                Nd4j.create(new double[][]{{1.0}, {0.0},
                        {1.0}, {0.0}}));
        Variable data2 = new Variable(
                Nd4j.create(new double[][]{{1.0, 0.0},
                        {1.0, 1.0},
                        {0.0, 1.0},
                        {0.0, 0.0}}));

        Variable target2 = new Variable(
                Nd4j.create(new double[][]{{1.0}, {0.0},
                        {1.0}, {0.0}}));
        Sequential nn = new Sequential(new Dense(2, 5),
                new ReLU(), new Dense(5, 1));
        Tensor pred2 = nn.predict(data);
        System.out.println(Arrays.toString(pred2.out.toDoubleVector()));
        RMSProp optimizer = new RMSProp(new Parameters(nn), 0.01f, 0.99f, 1e-8);
        for (int epoch = 1; epoch <= 500; epoch++) {
            MSELoss loss = Functional.mse_loss(nn.predict(data), target);
            loss.backward();
            optimizer.step();
        }
        for (int epoch = 1; epoch <= 500; epoch++) {
            MSELoss loss = Functional.mse_loss(nn.predict(data2), target2);
            loss.backward();
            optimizer.step();
        }
        Tensor pred = nn.predict(data);
        System.out.println(Arrays.toString(pred.out.toDoubleVector()));
        IntStream.rangeClosed(0, 3).forEach(i -> {
            assert Math.abs(target.out.getDouble(i) - pred.out.getDouble(i)) < 0.1;
        });
    }

}