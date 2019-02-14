package LibDL.Layer;

import LibDL.Tensor.Layer.Linear;
import LibDL.Tensor.Layer.MSE;
import LibDL.Tensor.Layer.ReLU;
import LibDL.Tensor.Layer.Sequential;
import LibDL.Tensor.Constant;
import LibDL.optim.SGD;
import org.nd4j.linalg.factory.Nd4j;
import org.junit.Test;

import java.util.stream.IntStream;

public class SequentialTest {

    @Test
    public void testSGD() {
        Constant data = new Constant(Nd4j.create(new double[][]{{1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.0, 0.0}}));
        Constant target = new Constant(Nd4j.create(new double[][]{{7.0}, {10.0}, {8.0}, {5.0}}));

        Sequential nn = new Sequential(new Linear(2, 1));
        nn.setX(data);

        MSE loss = new MSE(target);
        loss.setX(nn);

        loss.dout = Nd4j.create(new double[]{1.0});

        SGD optimizer = new SGD(nn.parameters(), 0.03f);

        for (int i = 1; i <= 800; i++) {
            loss.forward();
            loss.backward();
            optimizer.step();
        }

        IntStream.rangeClosed(0, 3).forEach(i->{
            assert Math.abs(target.value.getInt(i) - nn.out.getDouble(i)) < 0.001;
        });
    }

    @Test
    public void testMomentum() {
        Constant data = new Constant(Nd4j.create(new double[][]{{1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.0, 0.0}}));
        Constant target = new Constant(Nd4j.create(new double[][]{{7.0}, {10.0}, {8.0}, {5.0}}));

        Sequential nn = new Sequential(new Linear(2, 1));
        nn.setX(data);

        MSE loss = new MSE(target);
        loss.setX(nn);

        loss.dout = Nd4j.create(new double[]{1.0});

        SGD optimizer = new SGD(nn.parameters(), 0.03f, 0.8f);

        for (int i = 1; i <= 160; i++) {
            loss.forward();
            loss.backward();
            optimizer.step();
        }

        IntStream.rangeClosed(0, 3).forEach(i->{
            assert Math.abs(target.value.getInt(i) - nn.out.getDouble(i)) < 0.001;
        });
    }

    @Test
    public void testXOR() {
        Constant data = new Constant(Nd4j.create(new double[][]{{1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.0, 0.0}}));
        Constant target = new Constant(Nd4j.create(new double[][]{{1.0}, {0.0}, {1.0}, {0.0}}));

        Sequential nn = new Sequential(new Linear(2, 3), new ReLU(), new Linear(3, 1));
        nn.setX(data);

        MSE loss = new MSE(target);
        loss.setX(nn);

        loss.dout = Nd4j.create(new double[]{1.0});

        SGD optimizer = new SGD(nn.parameters(), 0.03f);

        for (int i = 1; i <= 800; i++) {
            loss.forward();
            loss.backward();
            optimizer.step();
        }

        IntStream.rangeClosed(0, 3).forEach(i->{
            assert Math.abs(target.value.getInt(i) - nn.out.getDouble(i)) < 0.1;
        });
    }
}