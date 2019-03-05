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

public class L1Regularize extends OperatorTensor {

    public L1Regularize(Constant[] params) {
        Stream<Constant> paramsStream = Arrays.stream(params);
        OperandInfo[] operandInfos = paramsStream
                .map(p -> new OperandInfo(p, () -> p.out.div(ND4JUtil.Abs(p.out))))
                .toArray(OperandInfo[]::new);

        Supplier<INDArray> forward = () -> paramsStream
                .map(p -> ND4JUtil.Abs(p.out).sum())
                .reduce(INDArray::addRowVector)
                .get();
        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }
}
