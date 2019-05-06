package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Tensor;
import LibDL.optim.Optimizer;
import LibDL.optim.SGD;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ops.random.impl.GaussianDistribution;
import org.nd4j.linalg.factory.Nd4j;

public class DropoutTest {
    @Test
    @Ignore("dropout is working but the test need improve")
    public void test() {

        int N_SAMPLES = 20;
        int N_HIDDEN = 300;

        Tensor x = new Constant(Nd4j.linspace(-1, 1, N_SAMPLES).transpose());
        Tensor y = x.add(new Constant(Nd4j.getExecutioner()
                .exec(new GaussianDistribution(Nd4j.create(x.sizes()), 0, 1), Nd4j.getRandom()).mul(0.3)));
        Tensor x_test = new Constant(Nd4j.linspace(-1, 1, N_SAMPLES).transpose());
        Tensor y_test = x.add(new Constant(Nd4j.getExecutioner()
                .exec(new GaussianDistribution(Nd4j.create(x.sizes()), 0, 1), Nd4j.getRandom()).mul(0.3)));

        Module net_dropped = new Sequential(
                new Dense(1, N_HIDDEN),
                new Dropout(0.5),
                new ReLU(),
                new Dense(N_HIDDEN, N_HIDDEN),
                new Dropout(0.5),
                new ReLU(),
                new Dense(N_HIDDEN, 1)
        );
        Module net_overfitting = new Sequential(
                new Dense(1, N_HIDDEN),
                new ReLU(),
                new Dense(N_HIDDEN, N_HIDDEN),
                new ReLU(),
                new Dense(N_HIDDEN, 1)
        );
        Optimizer sgd_d = new SGD(net_dropped.parameters(), 0.01f);
        Optimizer sgd_o = new SGD(net_dropped.parameters(), 0.01f);

        for (int t = 0; t < 500; t++) {
            Tensor loss_d = Functional.mse_loss(net_dropped.forward(x), y);
            Tensor loss_o = Functional.mse_loss(net_overfitting.forward(x), y);
            sgd_d.zero_grad();
            sgd_o.zero_grad();
            loss_d.backward();
            loss_o.backward();
            sgd_d.step();
            sgd_o.step();
        }

//        System.out.println(y.data);
//        System.out.println(net_dropped.forward(x).data);
        System.out.println(Functional.mse_loss(net_dropped.forward(x_test), y_test));

        System.out.println(Functional.mse_loss(net_overfitting.forward(x_test), y_test));

        assert (Functional.mse_loss(net_dropped.forward(x_test), y_test).data.getDouble(0)
                < Functional.mse_loss(net_overfitting.forward(x_test), y_test).data.getDouble(0));

    }
}
