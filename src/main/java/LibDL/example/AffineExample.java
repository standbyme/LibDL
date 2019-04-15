package LibDL.example;

import LibDL.Tensor.Variable;
import LibDL.nn.*;
import LibDL.optim.SGD;
import org.nd4j.linalg.factory.Nd4j;

import java.util.stream.IntStream;

public class AffineExample {

    public static void main(String[] args) {
        Variable data = new Variable(Nd4j.create(new double[][]{{1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.0, 0.0}}));
        Variable target = new Variable(Nd4j.create(new double[][]{{7.0}, {10.0}, {8.0}, {5.0}}));

        Sequential nn = new Sequential(new Dense(2, 1));
        nn.setInput(data);

        MSELoss loss = new MSELoss(target);
        loss.setInput(nn);

        SGD optimizer = new SGD(nn.parameters(), 0.3f, 0.8f);

        for (int epoch = 1; epoch <= 160; epoch++) {
            loss.forwardWithInput();
            loss.backward();
            optimizer.step();
        }

        IntStream.rangeClosed(0, 3).forEach(i -> {
            assert Math.abs(target.value.getInt(i) - nn.out.getDouble(i)) < 0.001;
        });
    }


}