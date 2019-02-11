package LibDL.Layer;

import LibDL.Tensor.Layer.Linear;
import LibDL.Tensor.Layer.MSE;
import LibDL.Tensor.Layer.Sequential;
import LibDL.Tensor.Constant;
import LibDL.optim.SGD;
import org.junit.Assert;
import org.nd4j.linalg.factory.Nd4j;
import org.junit.Test;

public class SequentialTest {

    @Test
    public void test() {
        Constant data = new Constant(Nd4j.create(new double[][]{{1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.0, 0.0}}));
        Constant target = new Constant(Nd4j.create(new double[][]{{7.0}, {10.0}, {8.0}, {5.0}}));

        Sequential nn = new Sequential(new Linear(2, 1), new MSE(target));

        nn.setX(data);
        nn.dout = Nd4j.create(new double[]{1.0});

        SGD optimizer = new SGD(nn.parameters(),0.03f);

        for (int i = 1; i <= 1000; i++) {
            nn.forward();
            nn.backward();
            optimizer.step();
        }

        Assert.assertEquals(nn.out.toString(), "2.5307e-10");
    }
}