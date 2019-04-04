package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.function.Supplier;


public class Sum extends OperatorTensor {

    private int[] dim;

    public Sum(Tensor tensor) {
        this(tensor, null);
    }

    public Sum(Tensor tensor, int... dimensions) {

        this.dim = dimensions;

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> {
                    long[] shape = new long[tensor.out.rank()];
                    INDArrayIndex[] indices = new INDArrayIndex[tensor.out.rank()];
                    int dimi = 0, douti = 0;
                    for (int i = 0; i < shape.length; i++) {
                        if (dimi < dim.length && dim[dimi] == i) {
                            shape[i] = 1;
                            dimi++;
                        } else shape[i] = dout.size(douti++);
                        indices[i] = NDArrayIndex.all();
                    }
                    INDArray ret = Nd4j.ones(shape);
                    ret.put(indices, dout);
                    return ret.broadcast(tensor.out.shape());
                })
        };

        Supplier<INDArray> forward = () -> {
            if (this.dim == null)
                this.dim = Nd4j.linspace(0, tensor.out.rank() - 1,
                        tensor.out.rank()).toIntVector();
            return tensor.out.sum(dim);
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}