package LibDL.nn;

import LibDL.Tensor.Module;
import LibDL.Tensor.Tensor;

public class Reshape extends Module {

    private long[] shape;

    public Reshape(long... shape) {
        this.shape = shape;
    }

    @Override
    public Tensor forward(Tensor input) {
        return (new LibDL.Tensor.Operator.Reshape(input, shape));
    }
}