package LibDL.Tensor;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

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
//        for (OperandInfo operandInfo : operatorInfo.operandInfos) {
//            if (operandInfo.tensor.requires_grad)
//                if (operandInfo.tensor.grad != null) {
//                    INDArray back = operandInfo.backward.get();
//                    operandInfo.tensor.grad = operandInfo.tensor.grad.broadcast(back).addi(back);
//                } else operandInfo.tensor.grad = operandInfo.backward.get();
//        }
//
//        for (OperandInfo operandInfo : operatorInfo.operandInfos) {
//            if (operandInfo.tensor.requires_grad) operandInfo.tensor.backward();
//        }
        ArrayList<Tensor> stack = new ArrayList<>();
        HashSet<Tensor> vis = new HashSet<>();
        dfs(this, vis, stack);
        for (int i = stack.size() - 1; i >= 0; i--) {
            Tensor tensor = stack.get(i);
            if (tensor instanceof OperatorTensor)
                ((OperatorTensor) tensor).backprop();
        }
    }

    private void backprop() {
        for (OperandInfo operandInfo : operatorInfo.operandInfos) {
            if (operandInfo.tensor.requires_grad)
                if (operandInfo.tensor.grad != null) {
                    operandInfo.tensor.grad.addi(operandInfo.backward.get());
                } else operandInfo.tensor.grad = operandInfo.backward.get();
        }
    }

    private static void dfs(Tensor now, Set<Tensor> vis, ArrayList<Tensor> stack) {
        vis.add(now);
        if (now instanceof OperatorTensor) {
            for (OperandInfo operandInfo : ((OperatorTensor) now).operatorInfo.operandInfos) {
                Tensor t = operandInfo.tensor;
                if (t.requires_grad && !vis.contains(t)) dfs(t, vis, stack);
            }
        }
        stack.add(now);
    }
}
