package LibDL.Tensor.Operator;

import LibDL.ND4JUtil;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

public class SoftmaxWithThread extends OperatorTensor {

    private static ExecutorService executor;

    public SoftmaxWithThread(Tensor tensor) {
        this(tensor, 0, false);
        System.err.println("UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.");
    }
    public SoftmaxWithThread(Tensor tensor, int dim) {
        this(tensor, dim, true);
    }

    private static INDArray DjSi(INDArray sn) {
        long n = sn.length();
        INDArray e = Nd4j.eye(n);
        INDArray right = sn.broadcast(n, n);
        INDArray left = sn.reshape(n, 1).broadcast(n, n);
        return left.muli(e.subi(right));
    }

    class ThreadSoftmax implements Runnable {
        private INDArray result;
        private long i;
        private INDArray out;
        private INDArray dout;
        ThreadSoftmax(INDArray result, long i, INDArray out, INDArray dout) {
            this.result = result;
            this.i = i;
            this.out = out;
            this.dout = dout;
        }
        @Override
        public void run() {
            result.putRow(i, dout.getRow(i).mmul(SoftmaxWithThread.DjSi(out.getRow(i))));
        }
    }

    private SoftmaxWithThread(Tensor tensor, int dim, boolean withDim) {

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
                        return dout.reshape(1, dout.shape()[0]).mmul(SoftmaxWithThread.DjSi(out.dup())).reshape(dout.shape()[0]);
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
                        INDArray result = Nd4j.zeros(size, lastDim);
                        executor = Executors.newCachedThreadPool();
                        for(long i = 0; i < size; i++) {
                            executor.execute(new ThreadSoftmax(result, i, out, dout));
                        }
                        executor.shutdown();
                        try {
                            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
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
