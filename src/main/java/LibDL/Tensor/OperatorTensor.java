package LibDL.Tensor;

import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public abstract class OperatorTensor extends Tensor {
    private OperatorInfo operatorInfo;

    protected void setOperatorInfo(OperatorInfo operatorInfo) {
        this.operatorInfo = operatorInfo;
        OperandInfo[] operandInfos = this.operatorInfo.operandInfos;

        for (OperandInfo operandInfo : operatorInfo.operandInfos) {
            if (operandInfo.tensor.requires_grad) {
                operandInfo.tensor.grad = Nd4j.emptyLike(operandInfo.tensor.data);
                operandInfo.tensor.outNumber++;
            }
        }

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
            if (operandInfo.tensor.requires_grad) {
                operandInfo.tensor.grad.addi(operandInfo.backward.get());
                operandInfo.tensor.outNumber--;
            }
        }

        for (OperandInfo operandInfo : operatorInfo.operandInfos) {
            if (operandInfo.tensor.requires_grad && operandInfo.tensor.outNumber == 0)
                operandInfo.tensor.backward();
        }
    }

    @Override
    public Variable[] parameters() {
        return Arrays.stream(operatorInfo.operandInfos)
                .flatMap(operandInfo -> Arrays.stream(operandInfo.tensor.parameters()))
                .toArray(Variable[]::new);
    }
}
