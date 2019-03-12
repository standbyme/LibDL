package LibDL.Tensor.Operator;

import LibDL.Tensor.Constant;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class main {
    @Test
    public void testSoftmax() {
        Constant data = new Constant(Nd4j.create(new double[]{0.3, 2.9, 4.0}));


        Softmax result = new Softmax(data);

        result.forward();

        double a = result.out.getDouble(0);
        double b = result.out.getDouble(1);
        double c = result.out.getDouble(2);

        assert a == 0.018211273476481438;
        assert b == 0.24519184231758118;
        assert c == 0.736596941947937;
    }

    @Test
    public void testCrossEntropyLoss() {
        Constant t = new Constant(Nd4j.create(new double[]{0.0, 0, 1, 0, 0, 0, 0, 0, 0, 0}));
        Constant y = new Constant(Nd4j.create(new double[]{0.1, 0.05, 0.6, 0, 0.05, 0.1, 0, 0.1, 0, 0}));


        CrossEntropyLoss result = new CrossEntropyLoss(y, t);
        result.forward();
        double a = result.out.getDouble(0);

        assert a == 0.5108253955841064;

        y = new Constant(Nd4j.create(new double[]{0.1, 0.05, 0.1, 0, 0.05, 0.1, 0, 0.6, 0, 0}));
        result = new CrossEntropyLoss(y, t);
        result.forward();

        double b = result.out.getDouble(0);

        assert b == 2.302584171295166;

    }

    @Test
    public void testUnfold() {

        Constant x = new Constant(Nd4j.linspace(1, 16, 16).reshape(2, 2, 2, 2));
        Unfold ret = new Unfold(x, 1, 2);
        INDArray im2colAssertion = Nd4j.create(new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 10.0,
                        0.0, 0.0, 0.0, 0.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.0, 14.0, 0.0, 0.0,
                        0.0, 0.0, 15.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                new int[]{2, 2, 1, 1, 6, 6});
        ret.forward();
        assertEquals(im2colAssertion, ret.out);

    }

    @Test
    public void testUnfold2() {

        INDArray assertion = Nd4j.create(new double[]{1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3,
                3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4,
                4, 4, 2, 2, 2, 2, 4, 4, 4, 4}, new int[]{1, 1, 2, 2, 4, 4});

        Constant x = new Constant(Nd4j.create(new double[]{1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                4, 4, 4, 4, 4, 4, 4, 4}, new int[]{1, 1, 8, 8}));

        Unfold ret = new Unfold(x, 2, 0, 2);

        ret.forward();
        assertEquals(assertion, ret.out);

    }
}
