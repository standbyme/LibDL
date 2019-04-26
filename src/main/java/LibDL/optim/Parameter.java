package LibDL.optim;

import LibDL.Tensor.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Parameter extends Variable {
    public Parameter(INDArray value) {
        super(value, true);
    }
}
