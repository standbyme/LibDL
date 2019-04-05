package LibDL.Tensor.Operator;

import LibDL.Tensor.Constant;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class SoftmaxTest {
    @Test
    public void testForward() {
        Constant data_to_forward;
        Softmax result;
        INDArray target;

        data_to_forward = new Constant(Nd4j.create(new double[]{0.3, 2.9, 4.0})); // 2D test
        result = new Softmax(data_to_forward, 1);
        result.forward();
        target = Nd4j.create(new double[] {0.01821127347648143768, 0.24519184231758117676, 0.73659694194793701172});
        assertEquals(target, result.out); // 1.4901161193847656E-8

        data_to_forward = new Constant(Nd4j.create(new double[][][]{
                {{4.3, 0.0, 2.0}, {-2., 1.0, 2.0}},
                {{4.1, 2.0, 2.0}, {0.0, 0.0, 1.2}}
        }));
        result = new Softmax(data_to_forward, -1);
        result.forward();
        target = Nd4j.create(new double[][][]{
                {{0.89780521392822265625, 0.01218192093074321747, 0.09001290053129196167}, {0.01321288757026195526, 0.26538792252540588379, 0.72139918804168701172}},
                {{0.80326908826828002930, 0.09836547076702117920, 0.09836547076702117920}, {0.18796579539775848389, 0.18796579539775848389, 0.62406843900680541992}}
        });
        assertEquals(target, result.out); // 9.685754776000977E-8

        result = new Softmax(data_to_forward, 1);
        result.forward();
        target = Nd4j.create(new double[][][]{
                {{0.99816703796386718750, 0.26894143223762512207, 0.50000000000000000000}, {0.00183293852023780346, 0.73105859756469726562, 0.50000000000000000000}},
                {{0.98369747400283813477, 0.88079702854156494141, 0.68997442722320556641}, {0.01630249992012977600, 0.11920291185379028320, 0.31002551317214965820}}
        });
        assertEquals(target, result.out); // 6.305829932292303E-10

        result = new Softmax(data_to_forward);
        result.forward();
        target = Nd4j.create(new double[][][]{
                {{0.54983407258987426758, 0.11920291185379028320, 0.50000000000000000000}, {0.11920291185379028320, 0.73105859756469726562, 0.68997442722320556641}},
                {{0.45016595721244812012, 0.88079702854156494141, 0.50000000000000000000}, {0.88079702854156494141, 0.26894143223762512207, 0.31002551317214965820}}
        });
        assertEquals(target, result.out); // 3.725290298461914E-9

        data_to_forward = new Constant(Nd4j.create(new double[] {0.3, 2.9, 4.0}).reshape(3));
        result = new Softmax(data_to_forward);
        result.forward();
        target = Nd4j.create(new double[] {0.01821127347648143768, 0.24519184231758117676, 0.73659694194793701172}).reshape(3);
        assertEquals(target, result.out); // 1.241763432820638E-9
    }

    @Test
    public void testBackward() {
        Constant data_to_backward;
        Softmax result;
        INDArray target;

        data_to_backward = new Constant(Nd4j.create(new double[] {1.0, 2.0, 3.0}).reshape(3), true);
        result = new Softmax(data_to_backward, 0);
        result.dout = Nd4j.create(new double[] { 2.18006110191345214844, -3.51054310798645019531, -4.66951799392700195312}).reshape(3);
        result.forward();
        result.backward();
        target = Nd4j.create(new double[] { 0.53561645746231079102,  0.06330370157957077026, -0.59892022609710693359});
        assertEquals(target, data_to_backward.dout); // 9.685754776000977E-8

        data_to_backward = new Constant(Nd4j.create(new double[][][] {
                {{1.0, 2.0, 3.0}, {-1.0, -2.0, 3.0}},
                {{5.0, 3.0, 3.1}, {1.5, 2.5, 3.5}}
        }), true);
        result = new Softmax(data_to_backward, 2);
        result.dout = Nd4j.create(new double[][][] {
                {{ 2.18006110191345214844, -3.51054310798645019531, -4.66951799392700195312}, { 2.03573584556579589844,  4.01314640045166015625,  7.95111751556396484375}},
                {{-8.44346332550048828125,  6.21065425872802734375, -5.96719074249267578125}, {-2.81993889808654785156, -4.51054286956787109375,  8.33048152923583984375}}
        });
        result.forward();
        result.backward();
        target = Nd4j.create(new double[][][] {
                {{ 0.53561645746231079102,  0.06330370157957077026, -0.59892022609710693359}, {-0.10334482789039611816, -0.02502040006220340729,  0.12836529314517974854}},
                {{-1.42557442188262939453,  1.35054612159729003906,  0.07502812147140502930}, {-0.63057214021682739258, -2.12781190872192382812,  2.75838351249694824219}}
        });
        assertEquals(target, data_to_backward.dout); // 1.1998539169629416E-7

        data_to_backward = new Constant(Nd4j.create(new double[][][] {
                {{1.0, 2.0, 3.0}, {-1.0, -2.0, 3.0}},
                {{5.0, 3.0, 3.1}, {1.5, 2.5, 3.5}}
        }), true);
        result = new Softmax(data_to_backward, 0);
        result.dout = Nd4j.create(new double[][][] {
                {{ 2.03597235679626464844, -3.46211719512939453125, -5.04995822906494140625}, { 2.15171647071838378906,  4.02197408676147460938,  6.75508117675781250000}},
                {{-8.03597259521484375000,  7.46211719512939453125, -5.15004158020019531250}, {-1.15171635150909423828, -3.02197384834289550781,  8.24491882324218750000}}
        });
        result.forward();
        result.backward();
        target = Nd4j.create(new double[][][] {
                {{ 0.17789781093597412109, -2.14783501625061035156,  0.02495841681957244873}, { 0.23158292472362518311,  0.07654115557670593262, -0.35011741518974304199}},
                {{-0.17789784073829650879,  2.14783453941345214844, -0.02495835907757282257}, {-0.23158286511898040771, -0.07654109597206115723,  0.35011735558509826660}}
        });
        assertEquals(target, data_to_backward.dout); // 6.317471464474995E-8
    }

}
