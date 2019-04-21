package LibDL.nn;

import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Tensor;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

public class Sequential extends LayerTensor {

    private final LayerTensor[] tensors;

    @JsonCreator
    public Sequential(@JsonProperty("tensors")LayerTensor... tensors) {
        this.tensors = tensors;
    }

    @Override
    protected Tensor core() {
        Tensor X = this.input;

        for (LayerTensor tensor : tensors) {
            tensor.setInput(X);
            X = tensor;
        }

        return X;
    }
}
