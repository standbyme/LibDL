package LibDL.optim;

import LibDL.Tensor.Constant;

public abstract class Optimizer {
    final Constant[] params;

    Optimizer(Constant[] params) {
        this.params = params;
    }

    public abstract void step();
}
