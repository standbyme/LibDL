package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.function.Supplier;

public class IndexSelect extends OperatorTensor {
    public IndexSelect(Tensor tensor, Tensor index) {
        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> Nd4j.zerosLike(tensor.data).put(index.data,this.grad))
        };

        Supplier<INDArray> forward = () -> tensor.data.get(index.data);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

}
