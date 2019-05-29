package LibDL;

import LibDL.TensorImpl.Nd4jTensor;

public interface Tensor {
    Nd4jTensor grad();

    Nd4jTensor detach();

    Nd4jTensor zero_();

    Nd4jTensor set_requires_grad(boolean requires_grad);

    boolean defined();

    int dim();

    @Override
    String toString();
}
