package LibDL.Tensor;

import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

public abstract class OperatorTensor extends Tensor {
    private OperatorInfo operatorInfo;

    protected void setOperatorInfo(OperatorInfo operatorInfo) {
        this.operatorInfo = operatorInfo;
        OperandInfo[] operandInfos = this.operatorInfo.operandInfos;

        // If operandInfos is empty, this line will panic
        requires_grad = Arrays.stream(operandInfos)
                .map(memberInfo -> memberInfo.tensor.requires_grad)
                .reduce(Boolean::logicalOr)
                .get();

        data = operatorInfo.forward.get();
    }

    @Override
    public final void backward() {
        if(data.length() == 1) // If this tensor is a scalar
            grad = Nd4j.create(new double[] {1.0});

        LinkedList<Tensor> tensorList = new LinkedList<>();
        traverse(this, new HashSet<>(), tensorList);

        for (Tensor tensor: tensorList) {
            if (tensor instanceof OperatorTensor)
                ((OperatorTensor) tensor).backprop();
        }
    }

    private void backprop() {
        for (OperandInfo operandInfo : operatorInfo.operandInfos) {
            if (operandInfo.tensor.requires_grad)
                if (operandInfo.tensor.grad != null) {
                    operandInfo.tensor.grad.addi(operandInfo.backward.get());
                } else
                    operandInfo.tensor.grad = operandInfo.backward.get();
        }
    }

    private static void traverse(Tensor current, Set<Tensor> visited, List<Tensor> nodeList) {
        visited.add(current);
        if (current instanceof OperatorTensor) {
            for (OperandInfo operandInfo : ((OperatorTensor) current).operatorInfo.operandInfos) {
                Tensor t = operandInfo.tensor;
                if (t.requires_grad && !visited.contains(t)) traverse(t, visited, nodeList);
            }
        }
        nodeList.add(0, current);
    }
}
