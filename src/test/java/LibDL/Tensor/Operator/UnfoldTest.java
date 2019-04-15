package LibDL.Tensor.Operator;


import LibDL.Tensor.Constant;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class UnfoldTest {
    @Test
    public void testUnfold() {
        Constant input = new Constant(Nd4j.linspace(1, 84, 84).reshape(1, 2, 6, 7));
        Unfold m = new Unfold.Builder(input, 3, 3).padding(2, 1).stride(2, 2).dilation(2, 1).build();
        m.forward();
        INDArray expected = Nd4j.create(new double[][][] {
                {{ 0.,  0.,  0.,  0.,  0.,  2.,  4.,  6.,  0., 16., 18., 20.},
                 { 0.,  0.,  0.,  0.,  1.,  3.,  5.,  7., 15., 17., 19., 21.},
                 { 0.,  0.,  0.,  0.,  2.,  4.,  6.,  0., 16., 18., 20.,  0.},
                 { 0.,  2.,  4.,  6.,  0., 16., 18., 20.,  0., 30., 32., 34.},
                 { 1.,  3.,  5.,  7., 15., 17., 19., 21., 29., 31., 33., 35.},
                 { 2.,  4.,  6.,  0., 16., 18., 20.,  0., 30., 32., 34.,  0.},
                 { 0., 16., 18., 20.,  0., 30., 32., 34.,  0.,  0.,  0.,  0.},
                 {15., 17., 19., 21., 29., 31., 33., 35.,  0.,  0.,  0.,  0.},
                 {16., 18., 20.,  0., 30., 32., 34.,  0.,  0.,  0.,  0.,  0.},
                 { 0.,  0.,  0.,  0.,  0., 44., 46., 48.,  0., 58., 60., 62.},
                 { 0.,  0.,  0.,  0., 43., 45., 47., 49., 57., 59., 61., 63.},
                 { 0.,  0.,  0.,  0., 44., 46., 48.,  0., 58., 60., 62.,  0.},
                 { 0., 44., 46., 48.,  0., 58., 60., 62.,  0., 72., 74., 76.},
                 {43., 45., 47., 49., 57., 59., 61., 63., 71., 73., 75., 77.},
                 {44., 46., 48.,  0., 58., 60., 62.,  0., 72., 74., 76.,  0.},
                 { 0., 58., 60., 62.,  0., 72., 74., 76.,  0.,  0.,  0.,  0.},
                 {57., 59., 61., 63., 71., 73., 75., 77.,  0.,  0.,  0.,  0.},
                 {58., 60., 62.,  0., 72., 74., 76.,  0.,  0.,  0.,  0.,  0.}}
        });
        assertEquals(expected, m.out);
    }
}
