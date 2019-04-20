package LibDL.example;


import LibDL.Tensor.Module;
import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import LibDL.nn.Dense;
import LibDL.nn.Functional;
import LibDL.nn.LossTensor;
import LibDL.nn.ReLU;
import LibDL.optim.Parameters;
import LibDL.optim.RMSProp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.stream.IntStream;

public class ModuleExample {

    private static class Model extends Module {
        private Module linear2_5, relu, linear5_1;

        Model() {
            linear2_5 = new Dense(2, 5);
            relu = new ReLU();
            linear5_1 = new Dense(5, 1);
        }

        @Override
        public Tensor forward(Tensor input) {
            // Still not good
            Tensor output = linear2_5.forward(input);
            output = relu.forward(output);
            output = linear5_1.forward(output);
            return output;
        }
    }

    public static void main(String[] args) {
        Variable data = new Variable(Nd4j.create(new double[][]{
                        {1.0, 0.0},
                        {1.0, 1.0},
                        {0.0, 1.0},
                        {0.0, 0.0}}));

        Variable target = new Variable(Nd4j.create(new double[][]{
                {1.0}, {0.0},
                {1.0}, {0.0}}));

        Model nn = new Model();
        RMSProp optimizer = new RMSProp(nn.parameters(), 0.01f, 0.99f, 1e-8);
        for (int epoch = 1; epoch <= 1000; epoch++) {
            Tensor output = nn.predict(data);
            LossTensor loss = Functional.mse_loss(output, target);
            loss.backward();
            optimizer.step();
        }
        Tensor pred = nn.predict(data);
        System.out.println(Arrays.toString(pred.out.toDoubleVector()));

        IntStream.rangeClosed(0, 3).forEach(i -> {
            assert Math.abs(target.value.getInt(i) - pred.out.getDouble(i)) < 0.1;
        });
    }
}
