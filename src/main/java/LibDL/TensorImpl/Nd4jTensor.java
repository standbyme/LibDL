package LibDL.TensorImpl;

import LibDL.Tensor;
import org.nd4j.autodiff.samediff.SDVariable;

public class Nd4jTensor implements Tensor {

    private SDVariable value;

    public Nd4jTensor(SDVariable value) {
        this.value = value;
    }

    @Override
    public Nd4jTensor grad() {
        return null;
    }

    @Override
    public Nd4jTensor detach() {
        return null;
    }

    @Override
    public Nd4jTensor zero_() {
        return null;
    }

    @Override
    public Nd4jTensor set_requires_grad(boolean requires_grad) {
        return null;
    }

    @Override
    public boolean defined() {
        return false;
    }

    @Override
    public int dim() {
        return 0;
    }

    @Override
    public String toString() {
        return value.getArr().toString();
    }
}
