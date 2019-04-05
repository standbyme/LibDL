package LibDL.nn;


import LibDL.Tensor.Operator.Log;
import LibDL.Tensor.Operator.NLLLoss;
import LibDL.Tensor.Operator.Softmax;
import LibDL.Tensor.Tensor;

public class CrossEntropyLoss extends LossTensor {

    private final Tensor target;

    public CrossEntropyLoss(Tensor target) {
        this.target = target;
    }

    @Override
    protected Tensor core() {
        return new NLLLoss.Builder(new Log(new Softmax(input)), target).reduction("mean").build();
    }

}