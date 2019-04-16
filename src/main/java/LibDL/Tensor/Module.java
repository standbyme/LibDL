package LibDL.Tensor;

public abstract class Module extends Tensor {

    protected InputTensor input;
    private Tensor core;

    public abstract Tensor forward(Tensor input);


    public void setInput(Tensor input) {
        this.input.setInput(input);
        core = forward(this.input);
        out = core.out;
        requires_grad = core.requires_grad;
    }

    protected Module() {
        input = new InputTensor();
    }

    @Override
    public void forwardWithInput() {
    }

    @Override
    public void backward() {
        core.dout = dout;
        core.backward();
    }

    public Tensor apply(Tensor input) {
        setInput(input);
        return this;
    }

    public Tensor predict(Tensor input) {        // make them happy
        return apply(input);
    }

    @Override
    public Variable[] parameters() {
        core = forward(this.input);
        return core.parameters();
    }

}
