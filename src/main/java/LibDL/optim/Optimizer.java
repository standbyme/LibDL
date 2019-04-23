package LibDL.optim;

import LibDL.Tensor.Variable;

public abstract class Optimizer {
    private final Parameters parameters;

    protected Variable[] params;

    Optimizer(Parameters parameters) {
        this.parameters = parameters;
    }

    final protected void cacheParams() {
        params = parameters.get();
    }

    public abstract void step();
}
