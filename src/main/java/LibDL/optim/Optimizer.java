package LibDL.optim;

import LibDL.Tensor.Variable;

public abstract class Optimizer {
    private final Parameters parameters;

    Variable[] params;

    Optimizer(Parameters parameters) {
        this.parameters = parameters;
    }

    final public void step() {
        params = parameters.get();
        step_core();
    }

    public abstract void step_core();
}
