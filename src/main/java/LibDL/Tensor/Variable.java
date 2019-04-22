package LibDL.Tensor;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Variable extends Tensor {

    public final INDArray value;

    public Variable(INDArray value) {
        this(value, false);
    }

    public Variable(INDArray value, Boolean requires_grad) {
        this.value = this.out = value;
        this.requires_grad = requires_grad;

    }

    @Override
    public void forwardWithInput() {
    }

    @Override
    public void backward() {

    }

    @Override
    public Variable[] parameters() {
        return requires_grad ? new Variable[]{this} : new Variable[]{};
    }
}
