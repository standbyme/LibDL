package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import java.util.Arrays;
import java.util.function.Supplier;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class MM extends OperatorTensor {

    public MM(Tensor mat1, Tensor mat2) {
        OperandInfo[] operandInfos = {
                new OperandInfo(mat1, () -> grad.mmul(mat2.data.transpose())),
                new OperandInfo(mat2, () -> mat1.data.transpose().mmul(grad)),
        };

        Supplier<INDArray> forward = () -> mat1.data.mmul(mat2.data);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

    public MM(Tensor mat1, Tensor mat2, boolean tensorMmul) {
        int rank = mat1.data.rank();

        OperandInfo[] operandInfos = {
                new OperandInfo(mat1, () -> tensorMmul(grad, mat2.data.transpose())),
                new OperandInfo(mat2, () -> {
                    INDArray X = mat1.data.reshape(-1, mat1.data.size(rank-1));
                    INDArray dZ = grad.reshape(-1, grad.size(rank-1));
                    return X.transpose().mmul(dZ);
                }),
        };

        Supplier<INDArray> forward = () -> tensorMmul(mat1.data, mat2.data);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

    private INDArray tensorMmul(INDArray tensor, INDArray mat) {
        long[] oldShape = tensor.shape();
        long cols = oldShape[oldShape.length-1];
        INDArray ret = tensor.reshape(-1, cols).mmul(mat);

        long[] newShape = Arrays.copyOf(oldShape, oldShape.length);
        newShape[newShape.length-1] = ret.size(1);
        return ret.reshape(newShape).dup();
    }
}