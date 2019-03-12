package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Tensor;
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

        assert loss.out.equalsWithEps(Nd4j.create(new double[][]{{92.5}}), 1e-6);

        loss.backward();

        assert x.dout.equalsWithEps(Nd4j.create(new double[][]{{30.0, 68.0}}), 1e-6);
        assert y.dout.equalsWithEps(Nd4j.create(new double[][]{{16.0, 22.0}, {8.0, 11.0}}), 1e-6);
        assert z.dout.equalsWithEps(Nd4j.create(new double[][]{{8.0, 11.0}}), 1e-6);
    }

}
