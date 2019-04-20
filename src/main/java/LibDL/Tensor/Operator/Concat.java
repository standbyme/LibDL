package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.Supplier;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;


public class Concat extends OperatorTensor {

    public Concat(Tensor... toConcat) {

        OperandInfo[] operandInfos = new OperandInfo[toConcat.length];

        final long size = toConcat[0].size(0);
        for(int i = 0; i < toConcat.length; i++) {
            assert toConcat[i].out.rank() == 2;

            final int fi = i;
            operandInfos[i] = new OperandInfo(toConcat[i],
                    () -> dout.get(interval(size*fi, size*fi+size), all()));
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
