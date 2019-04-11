package LibDL.Tensor;

public class InputTensor extends Tensor {
    Tensor inside;
    private boolean needs_forward;

    void setInput(Tensor input) {
        inside = input;
        requires_grad = input.requires_grad;
    }

    public void needsForward(boolean needForward) {
        needs_forward = needForward;
    }

    InputTensor() {
        inside = null;
        requires_grad = true;
        needs_forward = true;
    }

    @Override
    public void forward() {
        if (needs_forward) {
            inside.forward();
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
