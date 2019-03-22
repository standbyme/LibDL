package LibDL.nn;

import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Tensor;

public class ReLU extends LayerTensor {
    @Override
    protected Tensor core() {
        return new LibDL.Tensor.Operator.ReLU(input);
    }
}
