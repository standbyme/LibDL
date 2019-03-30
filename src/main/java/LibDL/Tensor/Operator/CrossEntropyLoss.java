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

                    System.out.println("magic" + Arrays.toString(magic.getRow(0).toDoubleVector()));

                    return magic;
                })
        };

        Supplier<INDArray> forward = () -> {
            INDArray y = tensor.out;
            INDArray t = target.value;

//            tao = t.sum(1).repeat(1,t.size(1));
            INDArray result = Transforms.log(y.add(delta));

            result = result.mul(t);
            result = result.sum(1);
            result = result.mul(-1.0);

            System.out.println("magic y " + Arrays.toString(y.getRow(0).toDoubleVector()));
//            System.out.println("magic t " +Arrays.toString(t.getRow(0).toDoubleVector()));

//            System.out.println("magic dout " +Arrays.toString(dout.getRow(0).toDoubleVector()));

            return result;
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
