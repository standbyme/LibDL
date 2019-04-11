package LibDL.optim;

import LibDL.Tensor.Variable;

public abstract class Optimizer {
    final Variable[] params;

    Optimizer(Variable[] params) {
        this.params = params;
    }

    public abstract void step();
}
