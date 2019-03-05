package LibDL.Tensor;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Supplier;

public class OperatorInfo {
    final OperandInfo[] operandInfos;
    final Supplier<INDArray> forward;

    public OperatorInfo(OperandInfo[] operandInfos, Supplier<INDArray> forward) {
        this.operandInfos = operandInfos;
        this.forward = forward;
    }
    
}
