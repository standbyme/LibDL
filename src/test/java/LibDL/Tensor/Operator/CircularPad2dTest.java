package LibDL.Tensor.Operator;

import LibDL.Tensor.Parameter;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

public class CircularPad2dTest {

    @Test
    public void testFxB() {
        INDArray x = Nd4j.linspace(0, 19, 20).reshape(1, 1, 5, 4);
        Parameter input = new Parameter(x);
        CircularPad2d m = new CircularPad2d(input, 3, 2, 3, 2);
        INDArray expected = Nd4j.create(new double[][][][]{{{
                { 9, 10, 11,  8,  9, 10, 11,  8,  9,},
                {13, 14, 15, 12, 13, 14, 15, 12, 13,},
                {17, 18, 19, 16, 17, 18, 19, 16, 17,},
                { 1,  2,  3,  0,  1,  2,  3,  0,  1,},
                { 5,  6,  7,  4,  5,  6,  7,  4,  5,},
                { 9, 10, 11,  8,  9, 10, 11,  8,  9,},
                {13, 14, 15, 12, 13, 14, 15, 12, 13,},
                {17, 18, 19, 16, 17, 18, 19, 16, 17,},
                { 1,  2,  3,  0,  1,  2,  3,  0,  1,},
                { 5,  6,  7,  4,  5,  6,  7,  4,  5,},
        }}});
        assertEquals(expected, m.data); // forward
        
        m.grad = Nd4j.onesLike(expected);
        m.backward();
        expected = Nd4j.create(new double[][][][]{
        {{{4., 6., 4., 4.},
          {4., 6., 4., 4.},
          {4., 6., 4., 4.},
          {4., 6., 4., 4.},
          {4., 6., 4., 4.}}}});
        assertEquals(expected, input.grad);
    }
}