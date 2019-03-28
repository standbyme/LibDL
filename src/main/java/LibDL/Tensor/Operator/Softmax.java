package LibDL.Tensor.Operator;

import LibDL.ND4JUtil;
import LibDL.Tensor.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

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
                    if(tensor.out.rank() == 1) {
                        return dout.mmul(Softmax.DjSi(out.dup()));
                    }else {
                        return null;
                    }
                })
        };

        Supplier<INDArray> forward = () -> {
            int rank = tensor.out.rank();
//            if(rank == 2 && tensor.out.tensorssAlongDimension(1) == 1) {
//                rank = 1;
//            }

            int _dim;
            if(withDim) {
                _dim = dim < 0 ? dim + rank : dim;
            }else {
                _dim = (rank == 1 || rank == 3) ? 0 : 1;
            }

//            if(rank == 1) {
//                Number max = tensor.out.maxNumber();
//                INDArray exp = ND4JUtil.Exp(tensor.out.sub(max));
//                Number sum = exp.sumNumber();
//                return exp.divi(sum);
//            }else {
            long[] shape = tensor.out.dup().shape();
            shape[_dim] = 1;

            INDArray maxAlongDim = tensor.out.max(_dim);
            maxAlongDim = maxAlongDim.reshape(shape).broadcast(tensor.out.shape());

            INDArray exp = ND4JUtil.Exp(tensor.out.sub(maxAlongDim));
            INDArray sumOfExpAlongDim = exp.sum(_dim);
            sumOfExpAlongDim = sumOfExpAlongDim.reshape(shape).broadcast(exp.shape());

            return exp.divi(sumOfExpAlongDim);
//            }
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
