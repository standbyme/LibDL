package LibDL.nn;

import LibDL.Tensor.Operator.Sum;
import LibDL.Tensor.Tensor;

public class MSELoss extends LossTensor {

    private final Tensor target;

    public MSELoss(Tensor target) {
        this.target = target;
    }

    @Override
    protected Tensor core() {
        return new Sum(X.sub(target).pow(2));
    }
}