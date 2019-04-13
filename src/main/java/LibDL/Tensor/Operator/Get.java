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
        assert i < tensor.out.size(0);

        INDArrayIndex[] indices = new INDArrayIndex[tensor.out.rank()];
        indices[0] = point(i);
        for(int i1 = 1; i1 < indices.length; i1++) {
            indices[i1] = all();
        }

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> {
                    INDArray result = Nd4j.zerosLike(tensor.out);
                    result.put(indices, dout);
                    return result;
                }),
        };

        Supplier<INDArray> forward = () -> tensor.out.get(indices);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

}
