package LibDL.Tensor;

import LibDL.Tensor.Operator.*;
import LibDL.nn.*;
import LibDL.nn.ReLU;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;


@JsonTypeInfo(
        use = JsonTypeInfo.Id.CLASS,
        include = JsonTypeInfo.As.PROPERTY,
        property = "type"
)
@JsonSubTypes({
        @JsonSubTypes.Type(value = Conv2d.class),
        @JsonSubTypes.Type(value = Linear.class),
        @JsonSubTypes.Type(value = MaxPool2d.class),
        @JsonSubTypes.Type(value = MSELoss.class),
        @JsonSubTypes.Type(value = ReLU.class),
        @JsonSubTypes.Type(value = Sequential.class),
        @JsonSubTypes.Type(value = SoftmaxWithLoss.class),
})
public abstract class Tensor implements Serializable {
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