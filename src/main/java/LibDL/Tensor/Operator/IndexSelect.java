package LibDL.Tensor.Operator;

import LibDL.ND4JUtil;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.function.Supplier;

public class IndexSelect extends OperatorTensor {
    public IndexSelect(Tensor tensor, Tensor index) {
        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> Nd4j.zerosLike(tensor.data).put(index.data, this.grad))
        };

        Supplier<INDArray> forward = () -> tensor.data.get(index.data);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }


    public IndexSelect(Tensor tensor, int dim, long... index) {
        final INDArrayIndex[] indArrayIndices = ND4JUtil.construct_indices_array((int) tensor.dim(), dim, index);

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> Nd4j.zerosLike(tensor.data).put(indArrayIndices, this.grad))
        };

        Supplier<INDArray> forward = () -> tensor.data.get(indArrayIndices);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

}
