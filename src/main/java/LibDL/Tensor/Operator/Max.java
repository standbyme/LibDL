package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.*;

import java.util.Arrays;
import java.util.function.Supplier;



public class Max extends OperatorTensor {

    private INDArray argmax;

    public Max(Tensor tensor) {

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> {
                    INDArray zeros = Nd4j.toFlattened(Nd4j.zerosLike(tensor.out));
                    INDArray indices = Nd4j.linspace(0, argmax.size(0) - 1, argmax.size(0));
                    indices = indices.mul(tensor.out.size(1)).add(argmax.transpose());
                    zeros.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.indices(indices.data().asLong())}, dout);
                    return zeros.reshape(tensor.out.shape());
                }),
        };

        Supplier<INDArray> forward = () -> {
            // returns max value of every row
            argmax = tensor.out.argMax(1);
            return tensor.out.max(1).transpose();
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}