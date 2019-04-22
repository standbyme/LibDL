package LibDL.optim;

import LibDL.Tensor.Variable;

public abstract class Optimizer {

    protected Variable[] params;

    Optimizer(Variable[] params) {
        this.params = params;
    }

    final protected void cacheParams() {
    }

    public abstract void step();
}
