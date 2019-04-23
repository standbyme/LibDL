package LibDL.optim;

import LibDL.Tensor.Variable;
import LibDL.nn.Module;

public class Parameters {
    private Module module;

    public Parameters(Module module) {
        this.module = module;
    }

    public Variable[] get() {
        return module.core.parameters();
    }
}
