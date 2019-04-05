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

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.function.Supplier;

public class SoftmaxWithoutLoop extends OperatorTensor {

    private static final Executor executor = Executors.newCachedThreadPool();

    public SoftmaxWithoutLoop(Tensor tensor) {
        this(tensor, 0, false);
        System.err.println("UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.");
    }
    public SoftmaxWithoutLoop(Tensor tensor, int dim) {
        this(tensor, dim, true);
    }

    private static INDArray DjSi(INDArray sn) {
        long n = sn.length();
        INDArray e = Nd4j.eye(n);
        INDArray right = sn.broadcast(n, n);
        INDArray left = sn.reshape(n, 1).broadcast(n, n);
        return left.muli(e.subi(right));
    }

    private SoftmaxWithoutLoop(Tensor tensor, int dim, boolean withDim) {

        OperandInfo[] operandInfos = { // TODO try to remove for-loop
                new OperandInfo(tensor, () -> {
                    int rank = out.rank();
                    int _dim;
                    if(withDim) {
                        _dim = dim < 0 ? dim + rank : dim;
                    }else {
                        _dim = (rank == 1 || rank == 3) ? 0 : 1;
                    }

                    if(tensor.out.rank() == 1) {
                        return dout.reshape(1, dout.shape()[0]).mmul(SoftmaxWithoutLoop.DjSi(out.dup())).reshape(dout.shape()[0]);
                    }else {
                        INDArray out = this.out.dup();
                        INDArray dout = this.dout.dup();

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
                        long lastDim = new Double(shape.getDouble(rank - 1)).longValue(); // TODO replace by TAD
                        shape.putScalar(rank - 1, 1);
                        long size = shape.prodNumber().longValue();
                        out = out.reshape(size, lastDim);
                        dout = dout.reshape(size, lastDim);



                        long rows = out.rows();
                        long cols = out.columns();
                        long rxc = rows * cols;

                        INDArray b1 = Nd4j.linspace(0, cols - 1, cols);
                        b1 = b1.transpose().broadcast(cols, rows).reshape(rxc);
                        INDArray indices1 = Nd4j.linspace(0, (rxc-1)*cols, rxc).reshape(rxc).addi(b1);
                        INDArray E = Nd4j.zeros(1, rxc * cols);
                        E.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.indices(indices1.data().asLong())},
                                Nd4j.ones(1, rxc));
                        E = E.reshape(cols, rxc);

                        INDArray left = out.transpose().reshape(1, rxc).transpose().broadcast(rxc, cols).reshape(cols, rxc);
                        INDArray right = out.reshape(1, rxc).broadcast(cols, rxc);
                        INDArray DjSi = left.muli(E.subi(right));

                        INDArray toShrink = dout.mmul(DjSi);

                        INDArray b2 = Nd4j.linspace(0, cols - 1, cols).broadcast(rows, cols);
                        INDArray indices2 = Nd4j.linspace(0, (rows+1)*cols*(rows-1), rows)
                                .broadcast(cols, rows).transpose().addi(b2).reshape(rxc);

                        INDArray result = toShrink.reshape(1, rxc * rows)
                                .get(NDArrayIndex.all(), NDArrayIndex.indices(indices2.data().asLong()))
                                .reshape(rows, cols);

                        result = result.reshape(__shape);
                        result.permutei(rearrange);
                        return result;
                    }
                })
        };

        Supplier<INDArray> forward = () -> {

            int rank = tensor.out.rank();
            int _dim;
            if(withDim) {
                _dim = dim < 0 ? dim + rank : dim;
            }else {
                _dim = (rank == 1 || rank == 3) ? 0 : 1;
            }

            long[] shape = tensor.out.dup().shape();
            shape[_dim] = 1;

            INDArray maxAlongDim = tensor.out.max(_dim);
            maxAlongDim = maxAlongDim.reshape(shape).broadcast(tensor.out.shape());

            INDArray exp = ND4JUtil.Exp(tensor.out.sub(maxAlongDim));
            INDArray sumOfExpAlongDim = exp.sum(_dim);
            sumOfExpAlongDim = sumOfExpAlongDim.reshape(shape).broadcast(exp.shape());

            return exp.divi(sumOfExpAlongDim);
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
