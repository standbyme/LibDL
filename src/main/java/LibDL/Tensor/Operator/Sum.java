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

    private static INDArray temp = Nd4j.zeros(1);

    public Sum(Tensor tensor) {
        this(tensor, null);
    }

    public Sum(Tensor tensor, int... dimensions) {

        this.dim = dimensions;

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> {
                    long[] shape = new long[tensor.data.rank()];
                    INDArrayIndex[] indices = new INDArrayIndex[tensor.data.rank()];
                    int dimi = 0, douti = 0;
                    for (int i = 0; i < shape.length; i++) {
                        if (dimi < dim.length && dim[dimi] == i) {
                            shape[i] = 1;
                            dimi++;
                        } else shape[i] = grad.size(douti++);
                        indices[i] = NDArrayIndex.all();
                    }
                    INDArray ret = Nd4j.ones(shape);
                    ret.put(indices, grad);
                    return ret.broadcast(tensor.data.shape());
//                    long[] shape_i = tensor.data.shape();
//                    long[] shape_t = temp.shape();
//
//                    if (shape_t.length != shape_i.length) {
//                        temp = Nd4j.zerosLike(tensor.data);
//                    }else {
//                        int n = shape_t.length;
//                        long[] expension = Arrays.copyOf(shape_t, n);
//                        for (int i = 0; i < n; i++) {
//                            if (shape_t[i] < shape_i[i]) {
//                                expension[i] = shape_i[i];
//                                System.out.print(Arrays.toString(temp.shape()) + " >>> ");
//                                temp = Nd4j.zeros(expension);
//                                System.out.println("new:" + i + "temp:shape:" + Arrays.toString(temp.shape()));
//                                shape_t = temp.shape();
//                            }
//                        }
//                    }
//
//                    INDArray result = temp.get(
//                            NDArrayIndex.interval(0, 1, shape_i[0]),
//                            NDArrayIndex.interval(0, 1, shape_i[1]),
//                            NDArrayIndex.interval(0, 1, shape_i[2]),
//                            NDArrayIndex.interval(0, 1, shape_i[3]),
//                            NDArrayIndex.interval(0, 1, shape_i[4]));
//
//                    Nd4j.getExecutioner().exec(new BroadcastCopyOp(tensor.data, ret, result, 0, 1, 3, 4));
//                    return result;
                })
        };

        Supplier<INDArray> forward = () -> {
            if (this.dim == null)
                this.dim = Nd4j.linspace(0, tensor.data.rank() - 1, tensor.data.rank()).toIntVector();
            return tensor.data.sum(dim);
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}