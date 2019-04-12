package LibDL.nn;

import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Tensor;

public class Sequential extends Module {

    private final Tensor[] tensors;

    public Sequential(Tensor... tensors) {
        this.tensors = tensors;
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor X = this.input;

        for (Tensor tensor : tensors) {
            ((LayerTensor) tensor).forward(X);
            X = tensor;
        }

        return X;
    }
}
