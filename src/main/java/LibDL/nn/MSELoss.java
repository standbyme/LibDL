package LibDL.nn;

import LibDL.Tensor.Operator.Sum;
import LibDL.Tensor.Tensor;

public class MSELoss extends LossLayer {

    private final Tensor target;
    private final boolean size_average;


    public MSELoss(Tensor target) {
        this(target, true);
    }

    public MSELoss(Tensor target, boolean size_average) {
        this.target = target;
        this.size_average = size_average;
    }

    @Override
    public Tensor apply(Tensor input) {
        if(size_average)
            return new Sum(input.sub(target).pow(2)).div((int) target.data.length());
        else
            return new Sum(input.sub(target).pow(2));
    }
}