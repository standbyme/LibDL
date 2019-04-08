package LibDL.Tensor.Operator;

import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.Supplier;


public class Sum extends OperatorTensor {

    private int[] dim;

    public Sum(Tensor tensor) {
        this(tensor, Nd4j.linspace(0, tensor.out.rank() - 1, tensor.out.rank()).toIntVector());
    }

    public Sum(Tensor tensor, int... dim) {

        this.dim = dim;

        OperandInfo[] operandInfos = {
                new OperandInfo(tensor, () -> Nd4j.onesLike(tensor.out).muli(dout)),
        };

        Supplier<INDArray> forward = () -> tensor.out.sum(dim);


        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}