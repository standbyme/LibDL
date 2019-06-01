package LibDL.TensorImpl;

import LibDL.Tensor;
import org.jetbrains.annotations.NotNull;
import org.nd4j.autodiff.samediff.SDVariable;

import java.util.Arrays;

public class Nd4jTensor implements Tensor {

    private @NotNull SDVariable variable;

    public @NotNull Nd4jTensor(@NotNull SDVariable variable) {
        this.variable = variable;
    }


    @Override
    public @NotNull Tensor add(@NotNull Tensor value) {
        assert value instanceof Nd4jTensor;
        return new Nd4jTensor(((Nd4jTensor) value).variable.add(variable));
    }

    @Override
    public @NotNull Tensor mm(@NotNull Tensor value) {
        assert value instanceof Nd4jTensor;
        return new Nd4jTensor(((Nd4jTensor) value).variable.mmul(variable));
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

        return variable.getArr().toString()
                + '\n'
                + "[ Variable"
                + Arrays.toString(variable.getShape()) +
                " ]";
    }
}
