package LibDL.nn;

import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Operator.Div;
import LibDL.Tensor.Operator.Exp;
import LibDL.Tensor.Operator.Sum;
import LibDL.Tensor.Tensor;

public class Softmax extends LayerTensor {

    private int dim;

    public Softmax(int dim) {
        this.dim = dim;
    }

    @Override
    protected Tensor core() {
        return new Div(new Exp(this.input), new Sum(this.input, 1));
    }
}