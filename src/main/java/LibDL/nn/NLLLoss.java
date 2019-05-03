package LibDL.nn;

import LibDL.Tensor.Tensor;

public class NLLLoss extends LossLayer {

    private final Tensor target;

    public NLLLoss(Tensor target) {
        this.target = target;
    }

    @Override
    public Tensor forward(Tensor input) {
        return new LibDL.Tensor.Operator.NLLLoss.Builder(input, target).build();
    }
}
