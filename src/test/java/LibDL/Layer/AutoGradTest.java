package LibDL.Layer;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Tensor;
import LibDL.nn.MSELoss;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;


public class AutoGradTest {

    @Test
    public void test() {
        Constant x = new Constant(Nd4j.create(new double[][] {{2.0, 1.0}}), true);
        Constant y = new Constant(Nd4j.create(new double[][] {{1.0, 2.0}, {3.0, 4.0}}), true);
        Constant z = new Constant(Nd4j.ones(1, 2).muli(3), true);

        Tensor out = x.mm(y).add(z);
        out.forward();

        assert out.out.equalsWithEps(Nd4j.create(new double[][]{{8.0, 11.0}}), 1e-6);

        MSELoss loss = new MSELoss(new Constant(Nd4j.zeros(1, 2)));
        loss.setX(out);
        loss.forward();

        assert loss.out.equalsWithEps(Nd4j.create(new double[][]{{185.0}}), 1e-6);

        loss.backward();

        assert x.dout.equalsWithEps(Nd4j.create(new double[][]{{60.0, 136.0}}), 1e-6);
        assert y.dout.equalsWithEps(Nd4j.create(new double[][]{{32.0, 44.0}, {16.0, 22.0}}), 1e-6);
        assert z.dout.equalsWithEps(Nd4j.create(new double[][]{{16.0, 22.0}}), 1e-6);
    }

}
