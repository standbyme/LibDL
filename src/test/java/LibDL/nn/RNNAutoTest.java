package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Parameter;
import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class RNNAutoTest {

    private static RNNAuto rnn = null;

    @BeforeClass
    public static void initRNN() {
        rnn = new RNNAuto(3, 5);
        rnn.weight_hh = new Parameter(Nd4j.create(new double[][]{
                {-0.2759, -0.2183, 0.4454, -0.0331, 0.0015},
                {-0.4196, -0.1691, -0.1807, 0.3347, -0.2393},
                {-0.3233, 0.4190, -0.3819, -0.1739, -0.2363},
                {-0.0749, 0.1831, 0.1638, 0.1701, 0.4200},
                {0.4375, -0.1900, -0.3810, -0.2224, -0.4320}}));

        rnn.weight_ih = new Parameter(Nd4j.create(new double[][]{
                {-0.0862, 0.1885, 0.1464},
                {-0.0782, -0.0145, -0.2172},
                {0.1067, 0.0100, -0.4008},
                {-0.0921, -0.1040, 0.1249},
                {0.0167, -0.1631, 0.1717}}));

        rnn.bias_hh = new Parameter(Nd4j.create(new double[][]{{-0.1757, -0.2823, -0.3362, -0.1846, -0.0046}}));

        rnn.bias_ih = new Parameter(Nd4j.create(new double[][]{{-0.1291, -0.2665, -0.0902, 0.3374, 0.2181}}));
    }

    @Test
    public void testRNNAuto() {
        Variable input = new Variable(Nd4j.create(new double[][][]{
                {{0.0053, -0.4601, -0.5414},
                        {-0.8522, 0.5896, 0.5357},
                        {-1.4926, 1.1163, -0.1855}},

                {{0.2956, 1.4636, 0.6051},
                        {1.0905, 0.5851, -0.4907},
                        {1.0333, -0.2276, -0.1532}}}
        ), true);

        Constant h0 = new Constant(Nd4j.create(new double[][]{
                {0.4765, 0.3493, -0.8491, 0.7070, 0.6665},
                {-0.3468, -0.0986, -0.5787, -0.6640, 0.5776},
                {-0.6707, 0.6514, 0.6557, 0.1384, -0.9628}}));

        INDArray output = Nd4j.create(new double[][][]{
                {{-0.7930, -0.4247, -0.1755, 0.3985, 0.2128},
                        {-0.1581, -0.6048, -0.4263, 0.2726, 0.1788},
                        {0.3239, -0.0783, -0.0573, 0.0470, -0.2979}},

                {{0.2493, -0.2030, -0.5348, 0.1577, -0.2880},
                        {-0.3656, -0.2373, -0.2324, -0.1165, 0.1217},
                        {-0.5072, -0.5500, -0.3009, -0.1026, 0.4916}}});

        Tensor result = rnn.forward(input, h0);

        assert result.data.equalsWithEps(output, 1e-3);


        result.grad = Nd4j.create(new double[][][]{{
                {1.1837e+00, 2.8680e-02, 5.9473e-01, -6.3787e-01, -9.8196e-01},
                {-1.5350e+00, 7.6189e-01, 1.3230e+00, -4.2295e-01, 5.3208e-01},
                {-1.4129e+00, -2.3161e+00, -2.5905e-02, 1.8038e+00, -7.0832e-01}},

                {{-1.7072e+00, -3.1917e+00, 1.1541e+00, -1.9135e+00, 2.3066e-01},
                        {-3.6231e-03, 3.9820e-01, 4.9735e-01, -1.5231e+00, -2.8920e-03},
                        {-4.8295e-01, -2.3305e+00, -1.2397e+00, 1.6851e+00, 1.8875e-01}}}
        );
        result.backward();


        INDArray inputGradient = Nd4j.create(new double[][][]{
                {{-0.0317, 0.5793, -0.3325},
                        {0.2363, -0.2091, -0.7278},
                        {0.1246, -0.2823, 0.4260}},

                {{0.6406, -0.0895, -0.0967},
                        {0.1593, 0.1554, -0.4589},
                        {-0.1132, -0.2520, 0.9854}}});
        INDArray weightGradient_ih = Nd4j.create(new double[][]{
                {1.0382, -4.0798, -2.3404},
                {0.6932, -6.3739, -1.5596},
                {-2.1144, 3.0991, 0.8036},
                {-2.5371, -1.6010, -0.3155},
                {-0.3486, 1.2805, 0.5369}});
        INDArray weightGradient_hh = Nd4j.create(new double[][]{
                {2.4463, 1.0570, 0.1811, 1.1793, -0.2149},
                {3.4779, 0.0099, -1.7617, -1.2541, 2.6618},
                {-1.9574, -0.2118, 0.0411, -0.1248, 0.0883},
                {0.4616, 2.1239, 3.8491, -1.7066, -4.3401},
                {-0.9795, -0.1644, 1.5444, -0.5361, -1.5196}});
        INDArray biasGradient = Nd4j.create(new double[]{-2.8803, -5.3557, 1.3418, -2.3531, -0.6663});

        assert input.grad.equalsWithEps(inputGradient, 1e-3);
        assert rnn.weight_ih.grad.equalsWithEps(weightGradient_ih, 1e-3);
        assert rnn.weight_hh.grad.equalsWithEps(weightGradient_hh, 1e-3);
        assert rnn.bias_ih.grad.equalsWithEps(biasGradient, 1e-3);
        assert rnn.bias_hh.grad.equalsWithEps(biasGradient, 1e-3);
    }
}
