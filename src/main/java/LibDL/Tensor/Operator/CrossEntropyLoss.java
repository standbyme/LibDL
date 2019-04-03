package LibDL.Tensor.Operator;

import LibDL.Tensor.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.function.Supplier;

public class CrossEntropyLoss extends OperatorTensor {

    private final static double delta = Math.pow(10, -7);

    public CrossEntropyLoss(Tensor tensor, Constant target) {


        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> {
                    INDArray magic = (tensor.out.sub(target.value));


                    return magic;
                })
        };

        Supplier<INDArray> forward = () -> {
            INDArray y = tensor.out;
            INDArray t = target.value;

            INDArray result = Transforms.log(y.add(delta)).muli(t).sum(1).muli(-1.0);


            return result;
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
