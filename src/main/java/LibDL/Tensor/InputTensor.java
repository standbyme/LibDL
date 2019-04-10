package LibDL.Tensor;

public class InputTensor extends Tensor {
    Tensor inside;

    void setInput(Tensor input) {
        inside = input;
        requires_grad = input.requires_grad;
    }

    InputTensor() {
        inside = null;
        requires_grad = true;
    }

    @Override
    public void forward() {
        inside.forward();
        this.out = inside.out;
    }

    @Override
    public void backward() {
        inside.dout = this.dout;
        inside.backward();
    }

    @Override
    public Constant[] parameters() {
        if (inside != null) {
            return inside.parameters();
        }
        return new Constant[]{};
    }
}
