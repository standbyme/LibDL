package LibDL.nn;

import LibDL.Tensor.LayerTensor;
import org.nd4j.linalg.factory.Nd4j;

abstract class LossTensor extends LayerTensor {
    LossTensor() {
        dout = Nd4j.create(new double[]{1.0});
//
    }

}
