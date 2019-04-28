package LibDL.nn;

import LibDL.Tensor.Tensor;

public class Sequential extends Module {

    private final Module[] layers;

    public Sequential(Module... layers) {
        this.layers = layers;
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor tensor = input;

        for (Module layer : layers) {
            tensor = layer.forward(tensor);
        }

        return tensor;
    }
}
