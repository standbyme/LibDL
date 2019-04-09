package LibDL.nn;

import LibDL.Tensor.LayerTensor;

public class Reshape extends LayerTensor {

    private long[] shape;

    public Reshape(long... shape) {
        this.shape = shape;
        setCore(new LibDL.Tensor.Operator.Reshape(input, shape));
    }

}