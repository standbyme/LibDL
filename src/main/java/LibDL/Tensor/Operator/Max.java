package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.function.Supplier;


public class Max extends OperatorTensor {

    private INDArray argmax;

    public Max(Tensor tensor) {

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> {
                    INDArray zeros = Nd4j.toFlattened(Nd4j.zerosLike(tensor.data));
                    INDArray indices = Nd4j.linspace(0, argmax.size(0) - 1, argmax.size(0),
                            DataType.LONG);
                    indices = indices.mul(tensor.data.size(1)).add(argmax);
                    zeros.put(new INDArrayIndex[]{NDArrayIndex.indices(indices.data().asLong())}, grad);
                    return zeros.reshape(tensor.data.shape());
                }),
        };

        Supplier<INDArray> forward = () -> {
            // returns max value of every row
            argmax = tensor.data.argMax(1);
            return tensor.data.max(1);
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}