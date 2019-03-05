package LibDL.Tensor;

import LibDL.Tensor.Operator.*;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Tensor {
    public INDArray out = null;
    public INDArray dout = null;

    boolean requires_grad;

    abstract public void forward();

    abstract public void backward();

    abstract public Constant[] parameters();

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

    final public Div div(int divisor) {
        return new Div(this, divisor);
    }
}