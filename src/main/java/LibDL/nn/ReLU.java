package LibDL.nn;

import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Tensor;

public class ReLU extends LayerTensor {
    public ReLU() {
        setCore(new LibDL.Tensor.Operator.ReLU(input));
    }
}
