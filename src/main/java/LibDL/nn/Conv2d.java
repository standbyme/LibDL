package LibDL.nn;

import LibDL.Tensor.Variable;
import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Operator.Unfold;
import org.nd4j.linalg.factory.Nd4j;

public class Conv2d extends LayerTensor {

    private final Variable W;
    private final Variable B;

    private final int kernel_size;
    private final int stride = 1;
    private final int padding = 0;
    private final boolean bias;

    public Conv2d(int kernel_size, boolean bias) {
//        super(false);
        W = new Variable(Nd4j.ones(1, kernel_size * kernel_size));
        B = new Variable(Nd4j.ones());

        this.kernel_size = kernel_size;
        this.bias = bias;

        Unfold col = new Unfold(input, kernel_size);

        if (bias) setCore(col.mm(W).add(B).reshapeLike(input));
        else setCore(col.mm(W).reshapeLike(input));
    }
}
