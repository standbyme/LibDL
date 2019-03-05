package LibDL.Tensor;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class OperandInfo {
    final Tensor tensor;
    final Supplier<INDArray> backward;

    public OperandInfo(Tensor tensor, Supplier<INDArray> backward) {
        this.tensor = tensor;
        this.backward = backward;
    }
}
