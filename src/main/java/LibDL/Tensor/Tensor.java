package LibDL.Tensor;

import LibDL.Tensor.Operator.*;
import LibDL.optim.Parameters;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Tensor {
    public INDArray out = null;
    public INDArray dout = null;

    private String tensorName;
    Tensor() {
        tensorName = this.getClass().getName();
    }

    public Tensor withName(String name) {
        tensorName = name;
        return this;
    }

    @Override
    public String toString() {
        return tensorName;
    }

    boolean requires_grad;

    abstract public void forwardWithInput();

    abstract public void backward();

    final public Parameters parameters(){
        return new Parameters(this);
    }

    public abstract Variable[] parameters_core();

    final public BroadcastAdd add(Tensor that) {
        return new BroadcastAdd(this, that);
    }

    final public Sub sub(Tensor that) {
        return new Sub(this, that);
    }

    final public MM mm(Tensor that) {
        return new MM(this, that);
    }

    final public Pow pow(int exponent) {
        return new Pow(this, exponent);
    }

    final public Max max() {
        return new Max(this);
    }

    final public Reshape reshapeLike(Tensor that) {
        return new Reshape(this, that.out.shape());
    }

    final public Reshape reshape(long... shape) {
        return new Reshape(this, shape);
    }

    final public Div div(int divisor) {
        return new Div(this, divisor);
    }

}