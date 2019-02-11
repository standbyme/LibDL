package LibDL.Tensor.Layer;

import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Tensor;

public class Sequential extends LayerTensor {

    private final LayerTensor[] tensors;

    public Sequential(LayerTensor... tensors) {
        this.tensors = tensors;
    }

    @Override
    protected Tensor core() {
        Tensor X = this.X;

        for (LayerTensor tensor : tensors) {
            tensor.setX(X);
            X = tensor;
        }

        return X;
    }
}
