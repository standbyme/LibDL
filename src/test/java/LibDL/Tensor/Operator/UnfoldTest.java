package LibDL.Tensor.Operator;


import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class UnfoldTest {
    @Test
    public void testUnfold() {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        Variable input = new Variable(Nd4j.linspace(1, 192, 192).reshape(2, 2, 8, 6), true);
        Unfold m = new Unfold.Builder(input, 2, 3).padding(2, 1).stride(2, 3).dilation(2, 1).build();
        INDArray expected = Nd4j.create(new double[][][]{
                {{0., 0., 0., 3., 0., 15., 0., 27., 0., 39.},
                        {0., 0., 1., 4., 13., 16., 25., 28., 37., 40.},
                        {0., 0., 2., 5., 14., 17., 26., 29., 38., 41.},
                        {0., 3., 0., 15., 0., 27., 0., 39., 0., 0.},
                        {1., 4., 13., 16., 25., 28., 37., 40., 0., 0.},
                        {2., 5., 14., 17., 26., 29., 38., 41., 0., 0.},
                        {0., 0., 0., 51., 0., 63., 0., 75., 0., 87.},
                        {0., 0., 49., 52., 61., 64., 73., 76., 85., 88.},
                        {0., 0., 50., 53., 62., 65., 74., 77., 86., 89.},
                        {0., 51., 0., 63., 0., 75., 0., 87., 0., 0.},
                        {49., 52., 61., 64., 73., 76., 85., 88., 0., 0.},
                        {50., 53., 62., 65., 74., 77., 86., 89., 0., 0.}},

                {{0., 0., 0., 99., 0., 111., 0., 123., 0., 135.},
                        {0., 0., 97., 100., 109., 112., 121., 124., 133., 136.},
                        {0., 0., 98., 101., 110., 113., 122., 125., 134., 137.},
                        {0., 99., 0., 111., 0., 123., 0., 135., 0., 0.},
                        {97., 100., 109., 112., 121., 124., 133., 136., 0., 0.},
                        {98., 101., 110., 113., 122., 125., 134., 137., 0., 0.},
                        {0., 0., 0., 147., 0., 159., 0., 171., 0., 183.},
                        {0., 0., 145., 148., 157., 160., 169., 172., 181., 184.},
                        {0., 0., 146., 149., 158., 161., 170., 173., 182., 185.},
                        {0., 147., 0., 159., 0., 171., 0., 183., 0., 0.},
                        {145., 148., 157., 160., 169., 172., 181., 184., 0., 0.},
                        {146., 149., 158., 161., 170., 173., 182., 185., 0., 0.}}
        });
        assertEquals(expected, m.data);
        m.grad = Nd4j.onesLike(m.data);
        m.backward();
        expected = Nd4j.create(new double[][][][]{
                {{{2., 2., 2., 2., 2., 0.},
                        {0., 0., 0., 0., 0., 0.},
                        {2., 2., 2., 2., 2., 0.},
                        {0., 0., 0., 0., 0., 0.},
                        {2., 2., 2., 2., 2., 0.},
                        {0., 0., 0., 0., 0., 0.},
                        {2., 2., 2., 2., 2., 0.},
                        {0., 0., 0., 0., 0., 0.}},

                        {{2., 2., 2., 2., 2., 0.},
                                {0., 0., 0., 0., 0., 0.},
                                {2., 2., 2., 2., 2., 0.},
                                {0., 0., 0., 0., 0., 0.},
                                {2., 2., 2., 2., 2., 0.},
                                {0., 0., 0., 0., 0., 0.},
                                {2., 2., 2., 2., 2., 0.},
                                {0., 0., 0., 0., 0., 0.}}},


                {{{2., 2., 2., 2., 2., 0.},
                        {0., 0., 0., 0., 0., 0.},
                        {2., 2., 2., 2., 2., 0.},
                        {0., 0., 0., 0., 0., 0.},
                        {2., 2., 2., 2., 2., 0.},
                        {0., 0., 0., 0., 0., 0.},
                        {2., 2., 2., 2., 2., 0.},
                        {0., 0., 0., 0., 0., 0.}},

                        {{2., 2., 2., 2., 2., 0.},
                                {0., 0., 0., 0., 0., 0.},
                                {2., 2., 2., 2., 2., 0.},
                                {0., 0., 0., 0., 0., 0.},
                                {2., 2., 2., 2., 2., 0.},
                                {0., 0., 0., 0., 0., 0.},
                                {2., 2., 2., 2., 2., 0.},
                                {0., 0., 0., 0., 0., 0.}}}});
        assertEquals(expected, input.grad);
    }
}
