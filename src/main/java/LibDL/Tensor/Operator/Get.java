package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import java.util.function.Supplier;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class Get extends OperatorTensor {

    public Get(Tensor tensor, long i) {
        assert i < tensor.data.size(0);

        INDArrayIndex[] indices = new INDArrayIndex[tensor.data.rank()];
        indices[0] = point(i);
        for (int i1 = 1; i1 < indices.length; i1++) {
            indices[i1] = all();
        }

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> {
                    INDArray result = Nd4j.zerosLike(tensor.data);
                    result.put(indices, grad);
                    return result;
                }),
        };

        Supplier<INDArray> forward = () -> tensor.data.get(indices);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

}
