package LibDL.nn;

import LibDL.Tensor.LayerTensor;
import com.fasterxml.jackson.annotation.JsonCreator;
import org.nd4j.linalg.factory.Nd4j;

abstract class LossTensor extends LayerTensor {
    @JsonCreator
    LossTensor() {
        dout = Nd4j.create(new double[]{1.0});
    }
}
