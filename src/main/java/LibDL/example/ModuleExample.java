package LibDL.example;


import LibDL.nn.Module;
import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import LibDL.nn.Dense;
import LibDL.nn.Functional;
import LibDL.nn.ReLU;
import LibDL.Tensor.Constant;
import LibDL.optim.RMSProp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.stream.IntStream;

public class ModuleExample {

    private static class Model extends Module {
        private Module fc1, relu, fc2;

        Model() {
            fc1 = new Dense(2, 5);
            relu = new ReLU();
            fc2 = new Dense(5, 1);
        }

        @Override
        public Tensor forward(Tensor input) {
            // Still not good
            Tensor output = fc1.forward(input);
            output = relu.forward(output);
            output = fc2.forward(output);
            return output;
        }
    }

    public static void main(String[] args) {
        Variable data = new Constant(Nd4j.create(new double[][]{
                        {1.0, 0.0},
                        {1.0, 1.0},
                        {0.0, 1.0},
                        {0.0, 0.0}}));

        Variable target = new Constant(Nd4j.create(new double[][]{
                {1.0}, {0.0},
                {1.0}, {0.0}}));

        Model nn = new Model();
        System.out.println(nn);

        RMSProp optimizer = new RMSProp(nn.parameters(), 0.01f, 0.99f, 1e-8);
        for (int epoch = 1; epoch <= 1000; epoch++) {
            optimizer.zero_grad();
            Tensor output = nn.forward(data);
            Tensor loss = Functional.mse_loss(output, target);
            loss.backward();
            optimizer.step();
        }
        Tensor pred = nn.forward(data);
        System.out.println(Arrays.toString(pred.data.toDoubleVector()));

        IntStream.rangeClosed(0, 3).forEach(i -> {
            assert Math.abs(target.data.getInt(i) - pred.data.getDouble(i)) < 0.1;
        });
    }
}
