package LibDL.nn;

import LibDL.Tensor.Operator.Sum;
import LibDL.Tensor.Tensor;

public class MSELoss extends LossLayer {

    public enum Reduction {
        mean,
        sum
    }

    private final Tensor target;
    private final Reduction reduction;


    public MSELoss(Tensor target) {
        this(target, Reduction.mean);
    }

    public MSELoss(Tensor target, Reduction reduction) {
        this.target = target;
        this.reduction = reduction;
    }

    @Override
    public Tensor forward(Tensor input) {
        switch (reduction) {
            case sum:
                return new Sum(input.sub(target).pow(2));
            case mean:
                return new Sum(input.sub(target).pow(2)).div((int) target.data.length());
            default:
                assert false;
        }
        return Tensor.ones(1);
    }
}