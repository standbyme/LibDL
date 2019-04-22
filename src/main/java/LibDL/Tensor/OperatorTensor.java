package LibDL.Tensor;

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

        out = operatorInfo.forward.get();
    }

    @Override
    public final void forwardWithInput() {

    }

    @Override
    public final void backward() {
        for (OperandInfo operandInfo : operatorInfo.operandInfos) {
            if (operandInfo.tensor.requires_grad) operandInfo.tensor.dout = operandInfo.backward.get();
        }

        for (OperandInfo operandInfo : operatorInfo.operandInfos) {
            if (operandInfo.tensor.requires_grad) operandInfo.tensor.backward();
        }
    }

    @Override
    public Variable[] parameters() {
        return Arrays.stream(operatorInfo.operandInfos)
                .flatMap(operandInfo -> Arrays.stream(operandInfo.tensor.parameters()))
                .toArray(Variable[]::new);
    }
}
