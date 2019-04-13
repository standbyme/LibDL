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

    final public Add add(Tensor that) {
        return new Add(this, that);
    }

    final public BroadcastAdd addVector(Tensor that) {
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

    final public Transpose transpose() {
        return new Transpose(this);
    }

    final public Tensor get(long i) {
        return new Get(this, i);
    }

    final public long size(int i) {
        return this.out.size(i);
    }
}