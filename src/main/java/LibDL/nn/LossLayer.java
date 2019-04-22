package LibDL.nn;

import LibDL.Tensor.Tensor;
import org.nd4j.linalg.factory.Nd4j;

public abstract class LossLayer extends Module {

    public Tensor eval(Tensor input) {
        Tensor loss = apply(input);
        loss.dout = Nd4j.create(new double[]{1.0});
        return loss;
    }

}
