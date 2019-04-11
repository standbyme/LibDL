package LibDL.nn;

import LibDL.Tensor.Variable;
import LibDL.Tensor.Operator.Sum;
import LibDL.Tensor.Tensor;

public class MSELoss extends LossTensor {

    private final Variable target;
    private final boolean size_average;


    public MSELoss(Variable target) {
        this(target, true);
    }

    public MSELoss(Variable target, boolean size_average) {
        this.target = target;
        this.size_average = size_average;
        setCore(core());
    }

    private Tensor core() {
        if(size_average)
            return new Sum(input.sub(target).pow(2)).div((int) target.value.length());
        else
            return new Sum(input.sub(target).pow(2));
    }
}