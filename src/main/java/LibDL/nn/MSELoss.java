package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Operator.Sum;
import LibDL.Tensor.Tensor;

public class MSELoss extends LossTensor {

    private final Constant target;
    private final boolean size_average;


    public MSELoss(Constant target) {
        this(target, true);
    }

    public MSELoss(Constant target, boolean size_average) {
        this.target = target;
        this.size_average = size_average;
    }

    @Override
    protected Tensor core() {
        if(size_average)
            return new Sum(X.sub(target).pow(2)).div((int) target.value.length());
        else
            return new Sum(X.sub(target).pow(2));
    }
}