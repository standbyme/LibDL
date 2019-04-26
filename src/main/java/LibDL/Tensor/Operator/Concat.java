package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;


public class Concat extends OperatorTensor {

    public Concat(Tensor... toConcat) {

        OperandInfo[] operandInfos = new OperandInfo[toConcat.length];

        final long size = toConcat[0].size(0);
        for(int i = 0; i < toConcat.length; i++) {
            assert toConcat[i].data.rank() == 2;

            final int fi = i;
            operandInfos[i] = new OperandInfo(toConcat[i],
                    () -> grad.get(interval(size*fi, size*fi+size), all()));
        }

        Supplier<INDArray> forward = () -> {
            INDArray[] valList = new INDArray[toConcat.length];
            for(int i = 0; i < toConcat.length; i++)
                valList[i] = toConcat[i].data;
            return Nd4j.concat(0, valList);
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

    /**
     * Self-concat
     * @param dim the dimension to be concat
     * @param times the times to be concat
     * */
    public Concat(Tensor input, int times, int dim) {
        OperandInfo[] operandInfos = {
                new OperandInfo(input, () -> {
                    long[] shape = new long[dout.shape().length + 1];
                    int i = 0;
                    for (; i < dim; i++) {
                        shape[i] = dout.shape()[i];
                    }
                    shape[i++] = times;
                    shape[i++] = dout.shape()[i - 2] / times;
                    for (; i < shape.length; i++) {
                        shape[i] = dout.shape()[i - 1];
                    }
                    return dout.reshape(shape).sum(dim);
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
