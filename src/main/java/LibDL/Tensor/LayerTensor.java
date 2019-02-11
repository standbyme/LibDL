package LibDL.Tensor;

public abstract class LayerTensor extends Tensor {

    protected Tensor X = null;

    public final void setX(Tensor x) {
        X = x;
        core = core();
        requires_grad = core.requires_grad;
    }

    private Tensor core;

    abstract protected Tensor core();

    @Override
    public void forward() {
        core.forward();
        out = core.out;
    }

    @Override
    public void backward() {
        core.dout = dout;
        core.backward();
    }

    @Override
    public Constant[] parameters() {
        return core.parameters();
    }
}
