package LibDL.Tensor;

public abstract class LayerTensor extends Tensor {

    protected InputTensor input;


    protected void setCore(Tensor core) {
        this.core = core;
        requires_grad = core.requires_grad;
    }

    public void setInput(Tensor input) {
        this.input.setInput(input);
    }


    protected LayerTensor() {
        input = new InputTensor();
    }


    public Tensor predict(Tensor input) {
        setInput(input);
        forwardThisLayer();
        return this;
    }

    private Tensor core;
    final public void forwardThisLayer() {
        input.needsForward(false);
        forward();
        input.needsForward(true);
    }

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
    public Variable[] parameters() {
        return core.parameters();
    }

    public Tensor forward(Tensor input) {
        setInput(input);
        return this;
    }
}
