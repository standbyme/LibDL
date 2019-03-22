package LibDL.Tensor.Operator;

import LibDL.Tensor.Constant;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertArrayEquals;
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

        Constant x1 = new Constant(Nd4j.linspace(0, 15, 16).reshape(4, 4));

        Unfold ret = new Unfold(x1, 3, 0, 1);
        ret.forward();

        INDArray assertion = Nd4j.create(new double[][]{{0, 1, 2, 4, 5, 6, 8, 9, 10}, {1, 2, 3, 5, 6, 7, 9, 10, 11}, {4, 5, 6, 8, 9, 10, 12, 13, 14}, {5, 6, 7, 9, 10, 11, 13, 14, 15}});

        assertEquals(assertion, ret.out);

    }


    @Test
    public void testReshape() {

        INDArray matrix = Nd4j.linspace(0, 5, 6);
        INDArray matrix_2_3 = matrix.reshape(2, 3);
        INDArray matrix_3_2 = matrix.reshape(3, 2);

        Constant x = new Constant(matrix_2_3, true);
        Reshape reshape = new Reshape(x, 3, 2);

        reshape.forward();
        assertArrayEquals(new long[]{2, 3}, reshape.from_shape);

        reshape.dout = matrix_3_2;
        reshape.backward();
        assertEquals(matrix_2_3, x.dout);

    }
}
