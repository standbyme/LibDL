package LibDL.Tensor;

public abstract class LayerTensor extends Tensor {

    protected Tensor X = null;

    final public void setX(Tensor x) {
        X = x;
        core = core();
        requires_grad = core.requires_grad;
    }

    private Tensor core;

    abstract protected Tensor core();

    @Override
    final public void forward() {
        core.forward();
        out = core.out;
    }

    @Override
    final public void backward() {
        core.dout = dout;
        core.backward();
    }

    @Override
    final public Constant[] parameters() {
        return core.parameters();
    }
}
