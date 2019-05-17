package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.*;

import java.util.function.Supplier;



public class Max extends OperatorTensor {

    private INDArray argMax;

    public Max(Tensor tensor) {

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> {
                    INDArray zeros = Nd4j.toFlattened(Nd4j.zerosLike(tensor.data));
                    INDArray indices = Nd4j.linspace(0, argMax.size(0) - 1, argMax.size(0));
                    indices = indices.mul(tensor.data.size(1)).add(argMax.transpose());
                    zeros.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.indices(indices.data().asLong())}, grad);
                    return zeros.reshape(tensor.data.shape());
                }),
        };

        Supplier<INDArray> forward = () -> {
            // returns max value of every row
            argMax = tensor.data.argMax(1);
            return tensor.data.max(1);
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

                    INDArray zeros = Nd4j.toFlattened(Nd4j.zerosLike(input.data));

                    long[] shape = input.data.permute(rearrange).shape();
                    long size = shape[0];
                    for (int i = 1; i < shape.length - 1; i++) {
                        size *= shape[i];
                    }

                    INDArray indices = Nd4j.linspace(0, (size - 1) * shape[input.data.rank() - 1], size);
                    indices.addi(Nd4j.toFlattened(argMax));
                    zeros.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.indices(indices.data().asLong())},
                            Nd4j.toFlattened(grad));
                    zeros = zeros.reshape(shape);
                    zeros.permutei(rearrange);
                    return zeros;
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