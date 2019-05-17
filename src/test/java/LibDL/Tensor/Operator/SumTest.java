package LibDL.Tensor.Operator;

import LibDL.Tensor.Parameter;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class SumTest {

    @Test
    public void testSum() {
        Parameter input = new Parameter(Nd4j.linspace(1, 48, 48).reshape(2, 2, 3, 2, 2));
        Sum m = new Sum(input, 2);

        INDArray expected = Nd4j.create(new double[][][][]{
                {
                        {{15, 18}, {21, 24}},
                        {{51, 54}, {57, 60}}
                },

                {
                        {{87, 90}, {93, 96}},
                        {{123, 126}, {129, 132}}
                }
        });
        assertEquals(expected, m.data);

        m.grad = Nd4j.linspace(1, 16, 16).reshape(2, 2, 2, 2);
        m.backward();
        assertEquals(Nd4j.create(new double[][][][]{ // notice reshape
             {{{    1.0000,    2.0000}, 
               {    3.0000,    4.0000}}, 
            
              {{    1.0000,    2.0000}, 
               {    3.0000,    4.0000}}, 
            
              {{    1.0000,    2.0000}, 
               {    3.0000,    4.0000}}}, 
            
            
             {{{    5.0000,    6.0000}, 
               {    7.0000,    8.0000}}, 
            
              {{    5.0000,    6.0000}, 
               {    7.0000,    8.0000}}, 
            
              {{    5.0000,    6.0000}, 
               {    7.0000,    8.0000}}}, 
            
            
             {{{    9.0000,   10.0000}, 
               {   11.0000,   12.0000}}, 
            
              {{    9.0000,   10.0000}, 
               {   11.0000,   12.0000}}, 
            
              {{    9.0000,   10.0000}, 
               {   11.0000,   12.0000}}}, 
            
            
             {{{   13.0000,   14.0000}, 
               {   15.0000,   16.0000}}, 
            
              {{   13.0000,   14.0000}, 
               {   15.0000,   16.0000}}, 
            
              {{   13.0000,   14.0000}, 
               {   15.0000,   16.0000}}}
        }).reshape(2, 2, 3, 2, 2), input.grad);
    }
}
