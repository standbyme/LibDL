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
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;


public class Concat extends OperatorTensor {

    public Concat(int dim, Tensor... toConcat) {
        int rank = toConcat[0].data.rank();
        OperandInfo[] operandInfos = new OperandInfo[toConcat.length];
        final Long[] sizes = Arrays.stream(toConcat)
                .map(tensor -> tensor.size(dim))
                .toArray(Long[]::new);

        long last = 0;
        for (int i = 0; i < toConcat.length; i++) {
            INDArrayIndex[] indices = new INDArrayIndex[rank];
            for(int d = 0; d < rank; d++) indices[d] = all();
            indices[dim] = interval(last, last + sizes[i]);
            operandInfos[i] = new OperandInfo(toConcat[i], () -> grad.get(indices));
            last = last + sizes[i];
        }

        Supplier<INDArray> forward = () ->
                Nd4j.concat(dim, Arrays.stream(toConcat)
                        .map(tensor -> tensor.data)
                        .toArray(INDArray[]::new));

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

    /**
     * Self-concat
     *
     * @param dim   the dimension to be concat
     * @param times the times to be concat
     */
    public Concat(Tensor input, int times, int dim) {
        OperandInfo[] operandInfos = {
                new OperandInfo(input, () -> {
                    long[] shape = new long[grad.shape().length + 1];
                    int i = 0;
                    for (; i < dim; i++) {
                        shape[i] = grad.shape()[i];
                    }
                    shape[i++] = times;
                    shape[i++] = grad.shape()[i - 2] / times;
                    for (; i < shape.length; i++) {
                        shape[i] = grad.shape()[i - 1];
                    }
                    return grad.reshape(shape).sum(dim);
                })
        };
        Supplier<INDArray> forward = () -> {
            INDArray[] toConcat = new INDArray[times];
            Arrays.fill(toConcat, input.data);
            return Nd4j.concat(dim, toConcat);
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);
        setOperatorInfo(operatorInfo);
    }

}
