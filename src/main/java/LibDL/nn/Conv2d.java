package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Operator.Unfold;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.factory.Nd4j;

public class Conv2d extends LayerTensor {

    private final Constant W;
    private final Constant B;

    private final int kernel_size;
    private final int stride = 1;
    private final int padding = 0;
    private final boolean bias;

    @Override
    protected Tensor core() {

        Unfold col = new Unfold(input, kernel_size);

        if (bias) return col.mm(W).addVector(B).reshapeLike(input);
        else return col.mm(W).reshapeLike(input);

    }

    public Conv2d(int kernel_size, boolean bias) {
        W = new Constant(Nd4j.ones(1, kernel_size * kernel_size));
        B = new Constant(Nd4j.ones());

        this.kernel_size = kernel_size;
        this.bias = bias;
    }
}
