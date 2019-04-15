package LibDL.Tensor;

public class Input extends Tensor {
    Tensor inside;
    private boolean needs_forward;

    void setInput(Tensor input) {
        inside = input;
        requires_grad = input.requires_grad;
    }

    public boolean isNotNull() {
        if (inside != null)
            return inside.out != null;
        return false;
    }

    public void needsForward(boolean needForward) {
        needs_forward = needForward;
    }

    Input() {
        inside = null;
        requires_grad = true;
        needs_forward = true;
    }

    @Override
    public void forwardWithInput() {
        if (needs_forward) {
            inside.forwardWithInput();
        }
        this.out = inside.out;
    }

    @Override
    public void backward() {
        inside.dout = this.dout;
        inside.backward();
    }

    @Override
    public Variable[] parameters() {
        if (inside != null) {
            return inside.parameters();
        }
        return new Variable[]{};
    }
}
