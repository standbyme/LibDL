package LibDL.Tensor;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Parameter extends Variable {
    public Parameter(INDArray value) {
        super(value, true);
    }
}
