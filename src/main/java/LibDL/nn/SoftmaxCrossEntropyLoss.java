package LibDL.nn;


import LibDL.Tensor.Operator.Log;
import LibDL.Tensor.Operator.NLLLoss;
import LibDL.Tensor.Operator.Softmax;
import LibDL.Tensor.Tensor;

public class SoftmaxCrossEntropyLoss extends LossTensor {

    private final Tensor target;

    public SoftmaxCrossEntropyLoss(Tensor target) {
        this.target = target;
    }

    @Override
    public Tensor forward(Tensor input) {
        return (new NLLLoss.Builder(new Log(new Softmax(input, 1)), target).reduction("mean").build());
    }

}