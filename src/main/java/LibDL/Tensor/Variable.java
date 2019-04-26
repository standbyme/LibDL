package LibDL.Tensor;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Variable extends Tensor {

    public Variable(INDArray value) {
        this(value, false);
    }

    public Variable(INDArray value, Boolean requires_grad) {
        this.data = value;
        this.requires_grad = requires_grad;
    }

    @Override
    public void backward() {
    }

}
