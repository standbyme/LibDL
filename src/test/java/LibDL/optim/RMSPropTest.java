package LibDL.optim;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import LibDL.nn.*;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.stream.IntStream;

public class RMSPropTest {

    @Test
    public void testRMSProp() {
        Variable data = new Variable(Nd4j.create(new double[][]{{1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.0, 0.0}}));
        Variable target = new Variable(Nd4j.create(new double[][]{{7.0}, {10.0}, {8.0}, {5.0}}));

        Sequential nn = new Sequential(
                new Dense(2, 5),
                new ReLU(),
                new Dense(5, 1));

        RMSProp optimizer = new RMSProp(nn.parameters(), 0.005f, 0.69f, 1e-8);
        for (int i = 1; i <= 1000; i++) {
            MSELoss loss = Functional.mse_loss(nn.predict(data), target);
            loss.backward();
            optimizer.step();
        }

        Tensor pred = nn.predict(data);
        IntStream.rangeClosed(0, 3).forEach(i -> {
            assert Math.abs(target.value.getInt(i) - pred.out.getDouble(i)) < 0.1;
        });
    }

}
