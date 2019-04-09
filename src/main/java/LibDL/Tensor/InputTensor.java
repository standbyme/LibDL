package LibDL.Tensor;

public class InputTensor extends Tensor {
    Tensor inside;

    void setInput(Tensor input) {
        inside = input;
        requires_grad = input.requires_grad;
    }

    InputTensor() {
        requires_grad = false;
    }

    @Override
    public void forward() {
        inside.forward();
        this.out = inside.out;
    }

    @Override
    public void backward() {
        inside.backward();
        this.dout = inside.dout;
    }

    @Override
    public Constant[] parameters() {
        if (requires_grad) return inside.parameters();
        return new Constant[]{};
    }
}
