package LibDL.nn;

import LibDL.Tensor.Tensor;

public class Reshape extends Module {

    private long[] shape;

    public Reshape(long... shape) {
        this.shape = shape;
    }

    @Override
    public Tensor apply(Tensor input) {
        return (new LibDL.Tensor.Operator.Reshape(input, shape));
    }
}