package LibDL.Tensor;

public abstract class Module extends Tensor {

    protected Input input;
    private Tensor core;
    //    private boolean requires_grad;
    private boolean coreIsSet;

    public abstract Tensor forward(Tensor input);

    private void checkCore() {
        if (coreIsSet) return;

        core = forward(input);
        coreIsSet = true;
        requires_grad = core.requires_grad;
    }

    public void setInput(Tensor input) {
        checkCore();
        this.input.setInput(input);
    }

    protected Module() {
        input = new Input();
        coreIsSet = false;
    }

    @Override
    public void forwardWithInput() {
        checkCore();
        core.forwardWithInput();
        out = core.out;
    }

    @Override
    public void backward() {
        checkCore();
        core.dout = dout;
        core.backward();
    }

    public Tensor apply(Tensor input) {
        checkCore();
        setInput(input);
        this.input.needsForward(false);
        forwardWithInput();
        this.input.needsForward(true);
        return this;
    }

    public Tensor predict(Tensor input) {        // make them happy
        return apply(input);
    }

    @Override
    public Variable[] parameters() {
        checkCore();
        return core.parameters();
    }

}
