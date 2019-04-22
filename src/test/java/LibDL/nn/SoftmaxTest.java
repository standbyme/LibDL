package LibDL.nn;

import LibDL.Tensor.Variable;
import LibDL.Tensor.Operator.Softmax;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.assertEquals;

public class SoftmaxTest {

    @Test
    public void testSoftmax() {

        Variable data = new Variable(Nd4j.create(new double[]{0.3, 2.9, 4.0}));
        Softmax f = new Softmax(data, 1);
        INDArray s = Transforms.abs(f.data.subi(Nd4j.create(new double[]{0.01821127329554753, 0.24519181293507392, 0.7365969137693786})), false);
        assert s.sumNumber().doubleValue() < 0.0000001;

//        assert a1 == 0.018211273476481438;
//        assert b1 == 0.24519184231758118;
//        assert c1 == 0.736596941947937;

        Variable data_to_forward = new Variable(Nd4j.create(new double[][][]{
                {{4.3, 0.0, 2.0}, {-2., 1.0, 2.0}},
                {{4.1, 2.0, 2.0}, {0.0, 0.0, 1.2}}
        }));
        Softmax result;
        INDArray target, loss;
        double lossSum;

        result = new Softmax(data_to_forward, data_to_forward.value.rank() - 1);
        target = Nd4j.create(new double[][][]{
                {{0.8978, 0.0122, 0.0900}, {0.0132, 0.2654, 0.7214}},
                {{0.8033, 0.0984, 0.0984}, {0.1880, 0.1880, 0.6241}}
        });
        loss = result.data.sub(target);
        lossSum = Transforms.abs(loss).sumNumber().doubleValue() / 12;
        assert lossSum / 12 < 0.000002;

        result = new Softmax(data_to_forward, 1);
        target = Nd4j.create(new double[][][]{
                {{0.9982, 0.2689, 0.5000}, {0.0018, 0.7311, 0.5000}},
                {{0.9837, 0.8808, 0.6900}, {0.0163, 0.1192, 0.3100}}
        });
        loss = result.data.sub(target);
        lossSum = Transforms.abs(loss).sumNumber().doubleValue() / 12;
        assert lossSum / 12 < 0.000002;

        result = new Softmax(data_to_forward);
        target = Nd4j.create(new double[][][]{
                {{0.5498, 0.1192, 0.5000}, {0.1192, 0.7311, 0.6900}},
                {{0.4502, 0.8808, 0.5000}, {0.8808, 0.2689, 0.3100}}
        });
        loss = result.data.sub(target);
        lossSum = Transforms.abs(loss).sumNumber().doubleValue() / 12;
        assert lossSum / 12 < 0.000002;

        data_to_forward = new Variable(Nd4j.create(new double[]{0.3, 2.9, 4.0}).reshape(3));
        result = new Softmax(data_to_forward);
        target = Nd4j.create(new double[]{0.0182, 0.2452, 0.7366}).reshape(3);
        loss = result.data.sub(target);
        lossSum = Transforms.abs(loss).sumNumber().doubleValue() / 12;
        assert lossSum / 3 < 0.000002;

        Variable data_to_backward; // The following is testing backward

        data_to_backward = new Variable(Nd4j.create(new double[]{1.0, 2.0, 3.0}).reshape(3), true);
        result = new Softmax(data_to_backward, 0);
        result.grad = Nd4j.create(new double[]{2.1801, -3.5105, -4.6695}).reshape(3);
        result.backward();
        assert Nd4j.create(new double[]
                {0.5356, 0.0633, -0.5989}).sub(data_to_backward.grad).sumNumber().doubleValue() < 0.0000005;

        data_to_backward = new Variable(Nd4j.create(new double[][][]{
                {{1.0, 2.0, 3.0}, {-1.0, -2.0, 3.0}},
                {{5.0, 3.0, 3.1}, {1.5, 2.5, 3.5}}
        }), true);
        result = new Softmax(data_to_backward, 2);
        result.grad = Nd4j.create(new double[][][]{
                {{2.1801, -3.5105, -4.6695}, {2.0357, 4.0131, 7.9511}},
                {{-8.4435, 6.2107, -5.9672}, {-2.8199, -4.5105, 8.3305}}
        });
        result.backward();
        assert Transforms.abs((Nd4j.create(new double[][][]{
                {{0.5356, 0.0633, -0.5989}, {-0.1033, -0.0250, 0.1284}},
                {{-1.4256, 1.3506, 0.0750}, {-0.6306, -2.1278, 2.7584}}
        }).sub(data_to_backward.grad))).sumNumber().doubleValue() / 12 < 0.0001; // TODO 1.3506 is 1.3505

        data_to_backward = new Variable(Nd4j.create(new double[][][]{
                {{1.0, 2.0, 3.0}, {-1.0, -2.0, 3.0}},
                {{5.0, 3.0, 3.1}, {1.5, 2.5, 3.5}}
        }), true);
        result = new Softmax(data_to_backward, 0);
        result.grad = Nd4j.create(new double[][][]{
                {{2.0360, -3.4621, -5.0500}, {2.1517, 4.0220, 6.7551}},
                {{-8.0360, 7.4621, -5.1500}, {-1.1517, -3.0220, 8.2449}}
        });
        result.backward();
        assert Transforms.abs((Nd4j.create(new double[][][]{
                {{0.1779, -2.1478, 0.0249}, {0.2316, 0.0765, -0.3501}},
                {{-0.1779, 2.1478, -0.0249}, {-0.2316, -0.0765, 0.3501}}
        }).sub(data_to_backward.grad))).sumNumber().doubleValue() / 12 < 0.0001; // TODO +-0.0249 should be +-0.0250
    }


}
