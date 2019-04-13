package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.Supplier;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;


public class Concat extends OperatorTensor {

    public Concat(Tensor... toConcat) {

        OperandInfo[] operandInfos = new OperandInfo[toConcat.length];

        for(int i = 0; i < toConcat.length; i++) {
            operandInfos[i] = new OperandInfo(toConcat[i], () -> dout.get(all()));
        }

        Supplier<INDArray> forward = () -> {
            INDArray[] valList = new INDArray[toConcat.length];
            for(int i = 0; i < toConcat.length; i++)
                valList[i] = toConcat[i].out;
            return Nd4j.concat(0, valList);
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);
    }

}
