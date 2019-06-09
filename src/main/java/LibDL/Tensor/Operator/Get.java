package LibDL.Tensor.Operator;

import LibDL.ND4JUtil;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import java.util.function.Supplier;

public class Get extends OperatorTensor {

    public Get(Tensor tensor, int dim, long i) {
        assert i < tensor.data.size(dim);

        INDArrayIndex[] indices = ND4JUtil.construct_indices_array((int) tensor.dim(), dim, i);
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

    public Get(Tensor tensor, long i) {
        this(tensor, 0, i);
    }

    public Get(Tensor tensor, int dim, long begin, long end) {
        INDArrayIndex[] indices = ND4JUtil.construct_chop_indices_array((int) tensor.dim(), dim, begin, end);
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
