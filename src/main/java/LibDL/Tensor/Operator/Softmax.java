package LibDL.Tensor.Operator;

import LibDL.ND4JUtil;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.Supplier;

public class Softmax extends OperatorTensor {

    public Softmax(Tensor tensor) {
        this(tensor, 0, false);
        System.err.println("UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.");
    }

    public Softmax(Tensor tensor, int dim) {
        this(tensor, dim, true);
    }

    private static INDArray DjSi(INDArray sn) {
        long n = sn.length();
        INDArray e = Nd4j.eye(n);
        INDArray right = sn.broadcast(n, n);
        INDArray left = sn.reshape(n, 1).broadcast(n, n);
        return left.muli(e.subi(right));
    }

    private Softmax(Tensor tensor, int dim, boolean withDim) {

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> {
                    int rank = data.rank();
                    int _dim;
                    if (withDim) {
                        _dim = dim < 0 ? dim + rank : dim;
                    } else {
                        _dim = (rank == 1 || rank == 3) ? 0 : 1;
                    }

                    if (tensor.data.rank() == 1) {
                        return grad.reshape(1, grad.shape()[0]).mmul(Softmax.DjSi(data.dup())).reshape(grad.shape()[0]);
                    } else {
                        INDArray out = this.data.dup();
                        INDArray dout = this.grad.dup();

                        int[] rearrange = Nd4j.linspace(0, rank - 1, rank).toIntVector();
                        rearrange[_dim] = rank - 1;
                        rearrange[rank - 1] = _dim;
                        out.permutei(rearrange);
                        dout.permutei(rearrange);

                        long[] __shape = out.shape();
                        double[] _shape = new double[rank];
                        for (int i = 0; i < rank; i++) {
                            _shape[i] = __shape[i];
                        }

                        INDArray shape = Nd4j.create(_shape);
                        long lastDim = new Double(shape.getDouble(rank - 1)).longValue();
                        shape.putScalar(rank - 1, 1);
                        long size = shape.prodNumber().longValue();
                        out = out.reshape(size, lastDim);
                        dout = dout.reshape(size, lastDim);
                        INDArray result = Nd4j.zeros(size, lastDim);
                        for (long i = 0; i < size; i++) {
                            result.putRow(i, dout.getRow(i, true).mmul(Softmax.DjSi(out.getRow(i))));
                        }
                        result = result.reshape(__shape);
                        result.permutei(rearrange);
                        return result;
                    }
                })
        };

        Supplier<INDArray> forward = () -> {

            int rank = tensor.data.rank();
            int _dim;
            if (withDim) {
                _dim = dim < 0 ? dim + rank : dim;
            } else {
                _dim = (rank == 1 || rank == 3) ? 0 : 1;
            }

            long[] shape = tensor.data.dup().shape();
            shape[_dim] = 1;

            INDArray maxAlongDim = tensor.data.max(_dim);
            maxAlongDim = maxAlongDim.reshape(shape).broadcast(tensor.data.shape());

            INDArray exp = ND4JUtil.Exp(tensor.data.sub(maxAlongDim));
            INDArray sumOfExpAlongDim = exp.sum(_dim);
            sumOfExpAlongDim = sumOfExpAlongDim.reshape(shape).broadcast(exp.shape());

            return exp.divi(sumOfExpAlongDim);
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
