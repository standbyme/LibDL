package LibDL.nn;

import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Tensor;

public class Softmax extends LayerTensor {

    private int dim;

    public Softmax(Tensor t) {
        this(t, 0);
    }

    public Softmax(int dim) {
        this.dim = dim;
    }

    public Softmax() {
        this(0);
    }

    public Softmax(Tensor t, int dim) {
        this(dim);
        this.setInput(t);
    }

    @Override
    protected Tensor core() {
        return new LibDL.Tensor.Operator.Softmax(this.input, dim);
//        return new Div(new Exp(this.input),
//                new Sum(new Exp(this.input), dim));
    }
}