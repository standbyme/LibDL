package LibDL.nn;

import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.factory.Nd4j;

abstract class LossTensor extends LayerTensor {
    LossTensor() {
        dout = Nd4j.create(new double[]{1.0});
    }

//    @Override
//    public void setInput(Tensor tensor) {
//        super.setInput(tensor);
//        this.forward();
//    }
}
