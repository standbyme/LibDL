package LibDL.Tensor;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

public abstract class OperatorTensor extends Tensor {
    private OperatorInfo operatorInfo;

    protected void setOperatorInfo(OperatorInfo operatorInfo) {
        this.operatorInfo = operatorInfo;
        OperandInfo[] operandInfos = this.operatorInfo.operandInfos;

//        System.out.println(Arrays.toString(operandInfos));
        // If operandInfos is empty, this line will panic
        requires_grad = Arrays.stream(operandInfos)
                .map(memberInfo -> memberInfo.tensor.requires_grad)
                .reduce(Boolean::logicalOr)
                .get();

        data = operatorInfo.forward.get();
    }

    @Override
    public final void backward() {
        for (OperandInfo operandInfo : operatorInfo.operandInfos) {
            if (operandInfo.tensor.requires_grad)
                if (operandInfo.tensor.grad != null) {
                    INDArray back = operandInfo.backward.get();
                    operandInfo.tensor.grad = operandInfo.tensor.grad.broadcast(back).addi(back);
                } else operandInfo.tensor.grad = operandInfo.backward.get();
        }

        for (OperandInfo operandInfo : operatorInfo.operandInfos) {
            if (operandInfo.tensor.requires_grad) operandInfo.tensor.backward();
        }
    }

}
