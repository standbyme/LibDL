package LibDL.nn;

import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Operator.Unfold;
import LibDL.Tensor.Tensor;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

public class MaxPool2d extends LayerTensor {

    private final int kernel_size;
    private final int stride = 1;
    private final int padding = 0;

    @JsonCreator
    public MaxPool2d(@JsonProperty("kernel_size")int kernel_size) {
        this.kernel_size = kernel_size;
    }

    @Override
    protected Tensor core() {
        Unfold col = new Unfold(input, kernel_size);
        return col.max().reshapeLike(input);
    }
}
