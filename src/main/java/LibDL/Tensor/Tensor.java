package LibDL.Tensor;

import LibDL.Tensor.Operator.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public abstract class Tensor {
    public INDArray data = null;
    public INDArray grad = null;

    private String tensorName;

    public Tensor() {
        tensorName = this.getClass().getSimpleName();
    }

    public Tensor withName(String name) {
        tensorName = name;
        return this;
    }

    @Override
    public String toString() {
        return tensorName + "(" + data + ")";
    }

    public boolean requires_grad;

    abstract public void backward();

    final public Add add(Tensor that) {
        return new Add(this, that);
    }

    final public Add add(Number rhs) {
        return new Add(this, rhs);
    }

    final public AddVector addVector(Tensor that) {
        return new AddVector(this, that);
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
        return new Reshape(this, that.data.shape());
    }

    final public Reshape reshape(long... shape) {
        return new Reshape(this, shape);
    }

    final public Tensor view(long... shape) {
        this.data.reshape(shape);
        return this;
    }

    final public Mul mul(Number times) {
        return new Mul(this, times);
    }

    final public Mul mul(Tensor times) {
        return new Mul(this, times);
    }

    final public Div div(Number divisor) {
        return new Div(this, divisor);
    }

    final public Transpose transpose() {
        return new Transpose(this);
    }

    final public Tensor get(long i) {
        return new Get(this, i);
    }

    final public long size(int i) {
        return this.data.size(i);
    }

    final public long dim() {
        return data.rank();
    }

    final public long[] sizes() {
        return data.shape();
    }

    public static Tensor zeros(long... shape) {
        return new Constant(Nd4j.zeros(shape));
    }

    public static Tensor ones(long... shape) {
        return new Constant(Nd4j.ones(shape));
    }

    public static Tensor numbers(Number number, long... shape) {
        Tensor r = new Constant(Nd4j.create(shape));
        r.data.assign(number);
        return r;
    }

}