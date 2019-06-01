package LibDL;

import org.jetbrains.annotations.NotNull;

public interface Tensor {
    @NotNull Tensor add(@NotNull Tensor value);

    @NotNull Tensor mm(@NotNull Tensor value);

    Tensor grad();

    Tensor detach();

    Tensor zero_();

    Tensor set_requires_grad(boolean requires_grad);

    boolean defined();

    int dim();

    @Override
    String toString();
}
