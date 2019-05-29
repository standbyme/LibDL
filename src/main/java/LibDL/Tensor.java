package LibDL;

public interface Tensor {
    Tensor grad();

    Tensor detach();

    Tensor zero_();

    Tensor set_requires_grad(boolean requires_grad);

    boolean defined();

    int dim();

    @Override
    String toString();
}
