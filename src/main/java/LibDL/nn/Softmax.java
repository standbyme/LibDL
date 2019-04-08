package LibDL.nn;

import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Tensor;

public class Softmax extends LayerTensor {

    private int dim;

    public Softmax(int dim) {
        this.dim = dim;
    }

    @Override
    protected Tensor core() {
        return new LibDL.Tensor.Operator.Softmax(input, dim);
    }
}