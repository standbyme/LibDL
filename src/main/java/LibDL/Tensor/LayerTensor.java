package LibDL.Tensor;

public abstract class LayerTensor extends Tensor {

    protected Tensor input = null;

    public void setInput(Tensor input) {
        this.input = input;
        core = core();
        requires_grad = core.requires_grad;
    }


    final public Tensor predict(Tensor input) {
        setInput(input);
        forward();

        return this;
    }

    private Tensor core;

    abstract protected Tensor core();

    @Override
    final public void forward() {
        core.forward();
        out = core.out;
    }

    @Override
    public void backward() {
        core.dout = dout;
        core.backward();
    }

    @Override
    final public Constant[] parameters() {
        return core.parameters();
    }
}
