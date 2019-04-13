package LibDL.nn;

import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;

public abstract class Module extends LayerTensor {

    private boolean coreIsSet;

    public Module() {
        coreIsSet = false;
    }

    @Override
    public void setInput(Tensor input) {
        checkCore();
        super.setInput(input);
    }

    private void checkCore() {
        if (coreIsSet) return;
        setCore(forward(this.input));
        coreIsSet = true;
    }

    @Override
    public Tensor predict(Tensor input) {
        checkCore();
        setInput(input);
        forwardThisLayer();
        return this;
    }

    @Override
    public Variable[] parameters() {
        checkCore();
        return super.parameters();
    }

//    abstract Tensor forward(Tensor input);
}
