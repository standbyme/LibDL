package LibDL.Tensor.Operator;

import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.function.Supplier;



public class Log extends OperatorTensor {
    public Log(Tensor tensor) {
        OperandInfo[] operandInfos = {
            new OperandInfo(tensor, () -> {
                double i = 1;
                System.out.println(dout);
                return Nd4j.create(new double[][] {{i, -2, i}, {-2, i, i}});
                // return Transforms.pow(this.dout.add(0.00001), -1, true);
            })
        };

        Supplier<INDArray> forward = () -> Transforms.log(tensor.out, true);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
