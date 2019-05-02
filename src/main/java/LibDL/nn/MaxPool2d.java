package LibDL.nn;

import LibDL.Tensor.Operator.Unfold;
import LibDL.Tensor.Tensor;

public class MaxPool2d extends Module {

    private final int kernel_size;
    private final int stride = 1;
    private final int padding = 0;

    public MaxPool2d(int kernel_size) {
        this.kernel_size = kernel_size;
    }

    @Override
    public Tensor forward(Tensor input) {
        return null;
    }
}
