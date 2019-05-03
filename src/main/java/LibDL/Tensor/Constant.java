package LibDL.Tensor;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Constant extends Variable {
    public Constant(INDArray value) {
        super(value, false);
    }
}
