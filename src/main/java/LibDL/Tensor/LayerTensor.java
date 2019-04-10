package LibDL.Tensor;

public abstract class LayerTensor extends Tensor {

    protected InputTensor input;


    protected void setCore(Tensor core) {
        this.core = core;
        requires_grad = core.requires_grad;
    }

    public void setInput(Tensor input) {
        this.input.setInput(input);
//        this.input.withName("("+this.input.toString()+")"+input.toString());
    }


    protected LayerTensor() {
        input = new InputTensor();
    }


    final public Tensor predict(Tensor input) {
        setInput(input);
//        forward();

        return this;
    }

    private Tensor core;

//    abstract protected Tensor core();

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
