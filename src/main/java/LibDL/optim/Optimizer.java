package LibDL.optim;

import LibDL.Tensor.Variable;
import org.nd4j.linalg.factory.Nd4j;

public abstract class Optimizer {

    protected Variable[] params;

    Optimizer(Variable[] params) {
        this.params = params;
    }

    public void zero_grad() {
        for (Variable param : params) {
            if (param.requires_grad) param.grad = Nd4j.zeros(1);
        }
    }

    public abstract void step();
}
