package LibDL.Tensor.Layer;

import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Operator.Sum;
import LibDL.Tensor.Tensor;

public class MSE extends LayerTensor {

    private final Tensor target;

    public MSE(Tensor target) {
        this.target = target;
    }

    @Override
    protected Tensor core() {
        return new Sum(X.sub(target).pow(2));
    }
}