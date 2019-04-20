package LibDL.nn;

import LibDL.Tensor.Module;
import org.nd4j.linalg.factory.Nd4j;

public abstract class LossTensor extends Module {
    LossTensor() {
        dout = Nd4j.create(new double[]{1.0});
    }

}
