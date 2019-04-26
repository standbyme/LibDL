package LibDL.nn;

import LibDL.Tensor.Tensor;
import LibDL.optim.Parameters;

public abstract class Module {

    public Tensor core;

    public abstract Tensor forward(Tensor input);

    public Tensor apply(Tensor input) {
        core = forward(input);
        return core;
    }

    public final Tensor predict(Tensor input) {        // make them happy
        return apply(input);
    }

    public Parameters parameters() {
        return new Parameters(this);
    }

}
