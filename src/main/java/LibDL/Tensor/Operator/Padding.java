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

public class Padding extends OperatorTensor {

    private static final double INF = Double.NEGATIVE_INFINITY;

    public Padding (Tensor input, int[] kernel_size, int[] padding, int[] stride, int[] dilation, boolean ceil_mode) {

        assert input.data.rank() == 4;
        assert input.data.shape()[1] == 1;

        OperandInfo[] operandInfos = new OperandInfo[] {
              new OperandInfo(input, null)
        };

        Supplier<INDArray> forward = () -> {
            long[] shape = input.data.dup().shape();

            long input_h = shape[2];
            long input_w = shape[3];

            long ceil_h = 0;
            long ceil_w = 0;
            if(ceil_mode) {
                int _filter_h = (kernel_size[0] - 1) * dilation[0] + 1;
                int _filter_w = (kernel_size[1] - 1) * dilation[1] + 1;
                long overage;
                overage= (input_h + padding[0] * 2 - _filter_h) % stride[0];
                if(overage != 0) {
                    long end = input_h + padding[0] * 2 + stride[0] - overage;
                    long begin = end - _filter_h + 1;
                    if(padding[0] == 0 || begin <= input_h + padding[0]) { // has primal data
                        ceil_h = stride[0] - overage;
                    }
                }
                overage= (input_w + padding[1] * 2 - _filter_w) % stride[1];
                if(overage != 0) {
                    long end = input_w + padding[1] * 2 + stride[1] - overage;
                    long begin = end - _filter_w + 1;
                    if(padding[1] == 0 || begin <= input_w + padding[1]) { // has primal data
                        ceil_w = stride[1] - overage;
                    }
                }
                System.out.println(ceil_h);
                System.out.println(ceil_w);
            }

            if(padding[0] > 0 || padding[1] > 0 || ceil_h > 0 || ceil_w > 0) {
                shape[2] = input_h + padding[0] * 2 + ceil_h;
                shape[3] = input_w + padding[1] * 2 + ceil_w;
                INDArray inf = Nd4j.zeros(shape).assign(INF);
                INDArrayIndex[] indArrayIndices;
                indArrayIndices = new INDArrayIndex[] {
                        NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(padding[0], padding[0] + input.data.shape()[2]),
                        NDArrayIndex.interval(padding[1], padding[1] + input.data.shape()[3])
                };
                inf.put(indArrayIndices, input.data);
                return inf;
            }
            return input.data;
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);
        setOperatorInfo(operatorInfo);
    }
}
