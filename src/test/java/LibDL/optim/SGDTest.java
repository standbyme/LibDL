package LibDL.optim;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Tensor;
import LibDL.nn.*;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.stream.IntStream;

public class SGDTest {

    @Test
    public void testSGD() {
        Constant data = new Constant(Nd4j.create(new double[][]{{1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.0, 0.0}}));
        Constant target = new Constant(Nd4j.create(new double[][]{{7.0}, {10.0}, {8.0}, {5.0}}));

        Sequential nn = new Sequential(new Dense(2, 1));
        SGD optimizer = new SGD(nn.parameters(), 0.3f, 0.8f);

        for (int epoch = 1; epoch <= 160; epoch++) {
            optimizer.zero_grad();
            Tensor loss = Functional.mse_loss(nn.predict(data), target);
            loss.backward();
            optimizer.step();
        }

        Tensor pred = nn.predict(data);
        IntStream.rangeClosed(0, 3).forEach(i -> {
            assert Math.abs(target.data.getDouble(i) - pred.data.getDouble(i)) < 0.001;
        });
    }


}