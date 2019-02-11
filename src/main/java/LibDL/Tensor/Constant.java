package LibDL.Tensor;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Constant extends Tensor {

    public final INDArray value;

    public Constant(INDArray value) {
        this(value, false);
    }

    public Constant(INDArray value, Boolean requires_grad) {
        this.value = value;
        this.requires_grad = requires_grad;
    }

    @Override
    public void forward() {
        out = value;
    }

    @Override
    public void backward() {

    }

    @Override
    public Constant[] parameters() {
        return requires_grad ? new Constant[]{this} : new Constant[]{};
    }
}
