package LibDL.Tensor.Operator;

import LibDL.ND4JUtil;
import LibDL.Tensor.Constant;
import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.function.Supplier;
import java.util.stream.Stream;

public class L2Regularize extends OperatorTensor {
    public L2Regularize(Constant[] params) {
        Stream<Constant> paramsStream = Arrays.stream(params);
        OperandInfo[] operandInfos = paramsStream
                .map(p -> new OperandInfo(p, () -> p.out.mul(2)))
                .toArray(OperandInfo[]::new);

        Supplier<INDArray> forward = () -> paramsStream
                .map(p -> ND4JUtil.pow(p.out, 2).sum())
                .reduce(INDArray::addRowVector)
                .get()
                .mul(0.5);

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
