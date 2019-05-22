package LibDL;

public class Tensor {

    private Num num;

    Tensor(Num num) {
        this.num = num;
    }

    public Tensor grad() {
        return null;
    }

    public Tensor detach() {
        return null;
    }

    public Tensor zero_() {
        return null;
    }

    public Tensor set_requires_grad(boolean requires_grad) {
        return null;
    }

    public boolean defined() {
        return false;
    }

    public int dim() {
        return 0;
    }

    @Override
    public String toString() {
        return null;
    }
}
