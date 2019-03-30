package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Operator.Log;
import LibDL.Tensor.Operator.Mul;
import LibDL.Tensor.Operator.Sum;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.factory.Nd4j;

public class CrossEntropy extends LossTensor {

    private final Constant target;
    private final boolean size_average;


    public CrossEntropy(Constant target) {
        this(target, true);
    }

    public CrossEntropy(Constant target, boolean size_average) {
        this.target = target;
        this.size_average = size_average;
    }

    @Override
    protected Tensor core() {
        return new Mul(new Sum(new Mul(new Log(this.input), this.target), 1),
                new Constant(Nd4j.create(new double[]{-1.0})));

    }
}
