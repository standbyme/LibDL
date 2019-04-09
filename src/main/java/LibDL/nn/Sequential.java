package LibDL.nn;

import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Tensor;

public class Sequential extends LayerTensor {

    private final Tensor[] tensors;

    public Sequential(Tensor... tensors) {
        this.tensors = tensors;
        setCore(core());
    }

    private Tensor core() {
        Tensor X = this.input;

        for (Tensor tensor : tensors) {
            ((LayerTensor) tensor).setInput(X);
            X = tensor;
        }

        return X;
    }
}
