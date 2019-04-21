package LibDL.nn;

import LibDL.Tensor.Module;
import LibDL.Tensor.Tensor;

public class Softmax extends Module {
    private int dim;

    public Softmax(int dim) {
        this.dim = dim;
    }

    public Softmax() {
        this(1);
    }

    @Override
    public Tensor forward(Tensor input) {
        return (new LibDL.Tensor.Operator.Softmax(input, dim));
    }

}