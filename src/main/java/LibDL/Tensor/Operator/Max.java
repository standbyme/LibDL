package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.bytedeco.javacpp.FloatPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.*;

import java.util.function.Supplier;



public class Max extends OperatorTensor {

    private INDArray argMax;

    public Max(Tensor input) {

        OperandInfo[] operandInfos = {
                new OperandInfo(input, () -> {
                    INDArray zeros = Nd4j.zerosLike(input.data).reshape(1, -1);
                    INDArray indices = Nd4j.linspace(0, argMax.size(0) - 1, argMax.size(0));
                    indices = indices.mul(input.data.size(1)).add(argMax.transpose());
                    zeros.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.indices(indices.data().asLong())}, grad);
                    return zeros.reshape(input.data.shape());
                }),
        };

        Supplier<INDArray> forward = () -> {
            // returns max value of every row
            argMax = input.data.argMax(1);
            return input.data.max(1);
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

    public Max(Tensor input, int dimension) {

        OperandInfo[] operandInfos = {
                new OperandInfo(input, () -> {

                    int rank = input.data.rank();
                    int[] rearrange = Nd4j.linspace(0, rank - 1, rank).toIntVector();
                    rearrange[dimension] = rank - 1;
                    rearrange[rank - 1] = dimension;

                    INDArray argMax = input.data.permute(rearrange).argMax(rank - 1);

                    INDArray zeros = Nd4j.zerosLike(input.data).reshape(1, -1);

                    long[] shape = input.data.permute(rearrange).shape();
                    long size = shape[0];
                    for (int i = 1; i < shape.length - 1; i++) {
                        size *= shape[i];
                    }

                    INDArray indices = Nd4j.linspace(0, (size - 1) * shape[input.data.rank() - 1], size);
                    indices.addi(argMax.reshape(1, -1));
                    float[] g = grad.reshape(1, -1).toFloatVector();
                    long[] i = indices.toLongVector();
                    FloatPointer floatPointer = (FloatPointer) zeros.data().pointer();
                    for (int j = 0; j < g.length; j++) {
                        floatPointer.position(i[j]).put(g, j, 1);
                    }
                    floatPointer.position(0);

                    return zeros.reshape(shape).permute(rearrange);
                }),
        };

        Supplier<INDArray> forward = () -> {
            argMax = input.data.argMax(dimension);
            return input.data.max(dimension);
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);
        setOperatorInfo(operatorInfo);
    }

    public INDArray getArgMax() {
        return argMax;
    }
}