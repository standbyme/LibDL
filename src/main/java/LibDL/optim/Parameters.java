package LibDL.optim;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;

public class Parameters {
    private Tensor tensor;

    public Parameters(Tensor tensor) {
        this.tensor = tensor;
    }

    public Variable[] get() {
        return tensor.parameters_core();
    }
}
