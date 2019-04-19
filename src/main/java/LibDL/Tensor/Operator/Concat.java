package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.function.Supplier;

public class Concat extends OperatorTensor {

    public Concat(Tensor input, int times, int dim) {
        OperandInfo[] operandInfos = {
            new OperandInfo(input, () -> {
                INDArrayIndex[] indArrayIndices = new INDArrayIndex[dout.rank()];
                Arrays.fill(indArrayIndices, NDArrayIndex.all());
                indArrayIndices[dim] = NDArrayIndex.interval(0, dout.shape()[dim] / times);
                return dout.get(indArrayIndices);
            })
        };
        Supplier<INDArray> forward = () -> {
            INDArray[] toConcat = new INDArray[times];
            Arrays.fill(toConcat, input.out);
            return Nd4j.concat(dim, toConcat);
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);
        setOperatorInfo(operatorInfo);
    }

}
