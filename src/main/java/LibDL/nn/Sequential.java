package LibDL.nn;

import LibDL.Tensor.Module;
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
            X = ((Module) tensor).forward(X);
        }

        return X;
    }
}
