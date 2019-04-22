package LibDL.example;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import LibDL.nn.*;
import LibDL.optim.SGD;
import org.nd4j.linalg.factory.Nd4j;


public class XORExample {

    public static void main(String[] args) {
        Variable data = new Variable(Nd4j.create(new double[][]{{1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.0, 0.0}}));
        Variable target = new Variable(Nd4j.create(new double[][]{{1.0}, {0.0}, {1.0}, {0.0}}));

        Sequential nn = new Sequential(new Dense(2, 5), new ReLU(), new Dense(5, 1));

        MSELoss criterion = new MSELoss(target);

        SGD optimizer = new SGD(nn.parameters(), 0.1f);

        for (int epoch = 1; epoch <= 1000; epoch++) {
            Tensor output = nn.predict(data);
            Tensor loss = criterion.eval(output);
            loss.backward();
            optimizer.step();
        }

        System.out.println(nn.predict(data).out);

    }

}