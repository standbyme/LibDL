package LibDL.nn;

import LibDL.Tensor.Tensor;

public class ReLU extends Module {
    @Override
    public Tensor apply(Tensor input) {
        return new LibDL.Tensor.Operator.ReLU(input);
    }
}
