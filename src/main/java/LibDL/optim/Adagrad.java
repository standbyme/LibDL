package LibDL.optim;

import LibDL.Tensor.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

public class Adagrad extends Optimizer {

    public Adagrad(Variable[] params) {
        super(params);
        throw new UnsupportedOperationException();
    }


    @Override
    public void step() {
    }
}
