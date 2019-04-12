package LibDL.nn;

import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Tensor;

public class Softmax extends LayerTensor {

    private int dim;

    public Softmax(int dim) {
        this.dim = dim;
        setCore(new LibDL.Tensor.Operator.Softmax(this.input, dim));
    }

    public Softmax() {
        this(0);
    }


}