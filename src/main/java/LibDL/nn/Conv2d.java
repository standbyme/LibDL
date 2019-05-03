package LibDL.nn;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Operator.Unfold;
import LibDL.Tensor.Parameter;
import org.nd4j.linalg.factory.Nd4j;

public class Conv2d extends Module {

    private final Parameter W;
    private final Parameter B;

    private final int kernel_size;
    private final int stride = 1;
    private final int padding = 0;
    private final boolean bias;

    public Conv2d(int kernel_size, boolean bias) {
        W = new Parameter(Nd4j.ones(1, kernel_size * kernel_size));
        B = new Parameter(Nd4j.ones());

        this.kernel_size = kernel_size;
        this.bias = bias;
    }

    @Override
    public Tensor forward(Tensor input) {
        Unfold col = new Unfold(input, kernel_size);
        if (bias) return (col.mm(W).addVector(B).reshapeLike(input));
        else return (col.mm(W).reshapeLike(input));
    }

}
