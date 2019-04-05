package LibDL.nn;

import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Tensor;

import java.util.Arrays;

public class Reshape extends LayerTensor {

    private long[] shape;

    public Reshape(long... shape) {
        this.shape = shape;
    }

    @Override
    protected Tensor core() {
        return new LibDL.Tensor.Operator.Reshape(input, shape);
    }
}