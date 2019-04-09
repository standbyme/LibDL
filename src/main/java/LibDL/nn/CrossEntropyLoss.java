package LibDL.nn;


import LibDL.Tensor.Operator.Log;
import LibDL.Tensor.Operator.NLLLoss;
import LibDL.Tensor.Operator.Softmax;
import LibDL.Tensor.Tensor;

public class CrossEntropyLoss extends LossTensor {

    private final Tensor target;

    public CrossEntropyLoss(Tensor target) {
        this.target = target;
        setCore(new NLLLoss.Builder(new Log(new Softmax(input, 1)), target).reduction("mean").build());
    }


}