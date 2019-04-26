package LibDL.optim;

import LibDL.Tensor.Variable;

public abstract class Optimizer {

    protected Variable[] params;

    Optimizer(Variable[] params) {
        this.params = params;
    }

    public void zero_grad() {
        for (Variable param : params) {
            if (param.grad != null)
                param.grad.assign(0);
        }
    }

    public abstract void step();
}
