package LibDL.Tensor;

public abstract class Module extends Tensor {

    protected InputTensor input;
    private Tensor core;
    //    private boolean requires_grad;
    private boolean coreIsSet;

    private void checkCore() {
        if (coreIsSet) return;
        setCore(forward(this.input));
    }


    protected void setCore(Tensor core) {
        this.core = core;
        this.coreIsSet = true;
        requires_grad = core.requires_grad;
    }

    public void setInput(Tensor input) {
        checkCore();
        this.input.setInput(input);
    }


    protected Module() {
        input = new InputTensor();
        coreIsSet = false;
    }


    final public void forward() {
        checkCore();
        input.needsForward(false);
        forwardWithInput();
        input.needsForward(true);
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


    public Tensor forward(Tensor input) {
        checkCore();
        setInput(input);
        return this;
    }

    public Tensor predict(Tensor input) {
        checkCore();
        setInput(input);
        forward();
        return this;
    }

    @Override
    public Variable[] parameters() {
        checkCore();
        return core.parameters();
    }

}
