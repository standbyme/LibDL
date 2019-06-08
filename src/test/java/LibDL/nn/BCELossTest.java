package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class BCELossTest {

    @Test
    public void test() {
        {
            BCELoss bceLoss = new BCELoss();
            Variable x = new Variable(Nd4j.create(new double[]{0.235, 0.856, 0.312}), true);
            Tensor loss = bceLoss.forward(x, new Constant(Nd4j.create(new double[]{
                    1, 0, 1
            })));
            System.out.println(loss);
            loss.backward();
            System.out.println(x.grad);
            assertEquals(Nd4j.create(new double[]{1.51695454120635986328}).reshape(1), loss.data);
            assertEquals(Nd4j.create(new double[]{
                    -1.41843974590301513672, 2.31481480598449707031, -1.06837606430053710938
            }), x.grad);
        }
        {
            BCELoss bceLoss = new BCELoss(new Constant(Nd4j.create(new double[]{1, -2, 3})), "sum");
            Variable x = new Variable(Nd4j.create(new double[]{0.235, 0.856, 0.312}), true);
            Tensor loss = bceLoss.forward(x, new Constant(Nd4j.create(new double[]{
                    1, 0, 1
            })));
            System.out.println(loss);
            loss.backward();
            System.out.println(x.grad);
            assertEquals(Nd4j.create(new double[]{1.06654202938079833984}).reshape(1), loss.data);
            assertEquals(Nd4j.create(new double[]{
                    -4.25531911849975585938, -13.88888931274414062500, -9.61538410186767578125
            }), x.grad);
        }
    }
}
