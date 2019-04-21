package LibDL.nn;

import LibDL.Tensor.Operator.CrossEntropyLoss;
import LibDL.Tensor.Operator.Softmax;
import LibDL.Tensor.Tensor;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

public class SoftmaxWithLoss extends LossTensor {

    private final Tensor target;
    private Tensor y;

    @JsonCreator
    public SoftmaxWithLoss(@JsonProperty("target")Tensor target) {
        this.target = target;
    }

    @Override
    protected Tensor core() {
        y = new Softmax(input);
        return new CrossEntropyLoss(y,target);
    }

    @Override
    public void backward() {
        long batch_size = target.out.shape()[0];
        input.dout = y.out.sub(target.out).divi(batch_size);
        input.backward();
    }
}