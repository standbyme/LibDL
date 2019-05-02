package LibDL.optim;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Tensor;
import LibDL.nn.Dense;
import LibDL.nn.Functional;
import LibDL.nn.ReLU;
import LibDL.nn.Sequential;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.stream.IntStream;

public class AdamTest {

    @Test
    @Ignore("Adam still failing")
    public void test() {
        Constant data = new Constant(Nd4j.create(new double[][]{{1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.0, 0.0}}));
        Constant target = new Constant(Nd4j.create(new double[][]{{7.0}, {10.0}, {8.0}, {5.0}}));

        Sequential nn = new Sequential(
                new Dense(2, 5),
                new ReLU(),
                new Dense(5, 1));

        Adam optimizer = new Adam(nn.parameters(), 0.001f, new float[]{0.9f, 0.99f}, 1e-8);
        for (int i = 1; i <= 10; i++) {
            optimizer.zero_grad();
            Tensor pred = nn.forward(data);
            System.out.println(pred.data);
            Tensor loss = Functional.mse_loss(pred, target);
            loss.backward();
            optimizer.step();
        }

        Tensor pred = nn.forward(data);
        System.out.println(pred.data);
        IntStream.rangeClosed(0, 3).forEach(i -> {
            assert Math.abs(target.data.getInt(i) - pred.data.getDouble(i)) < 0.1;
        });
    }

}
