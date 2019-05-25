package LibDL.Tensor.Operator;

import LibDL.Tensor.*;
import LibDL.utils.INDArrayPointer;
import org.bytedeco.javacpp.FloatPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.function.Supplier;

public class Correlation extends OperatorTensor {

    private static INDArrayPointer temp1 = new INDArrayPointer("temp1");
    private static INDArrayPointer temp2 = new INDArrayPointer("temp2");
    private static INDArrayPointer temp3 = new INDArrayPointer("temp3");
    private static INDArrayPointer temp4 = new INDArrayPointer("temp4");

    Correlation(Tensor input, Tensor weight, long ah, long aw, int ic, int oc, int groups) {

        assert input.data.rank() == 3;
        assert weight.data.rank() == 3;
        assert weight.data.shape()[0] == 1 && weight.data.shape()[2] == 1;
        assert ic % groups == 0;
        assert oc % groups == 0;

        long[] shape_i = input.data.shape();
        long[] shape_w = weight.data.shape();

        long N = shape_i[0];
        long L = shape_i[2];
        long HxW = shape_w[1] / oc / ic;
        long sxi = shape_i[1]; // size * ic
        long size = sxi / ic;

        OperandInfo[] operandInfos = new OperandInfo[]{
                new OperandInfo(input, () -> {

                    INDArray rst2 = Nd4j.zeros(sxi, N, L);

                    INDArray rst3 = Nd4j.zeros(sxi, N, L);
                    FloatPointer pointerRst3 = (FloatPointer) rst3.data().pointer();

                    INDArray rst4 = temp2.expandAndReturnTemp(N, 1, L);
                    int len = (int) N * (int) L;
                    FloatPointer pointerRst4;
                    float[] floatsRst4 = new float[len];

                    for (int j = 0; j < oc; j++) {
                        for (int i = 0; i < sxi; i++) {
                            Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(
                                    grad.reshape(N, oc, L).get(NDArrayIndex.all(), NDArrayIndex.interval(j, j + 1), NDArrayIndex.all()),
                                    weight.data.get(NDArrayIndex.all(), NDArrayIndex.interval(j * sxi + i , j * sxi + i + 1), NDArrayIndex.all()),
                                    rst4,
                                    1));
                            pointerRst4 = (FloatPointer) rst4.data().pointer();
                            pointerRst4.get(floatsRst4, 0, len);
                            pointerRst3.position(i * len);
                            pointerRst3.put(floatsRst4, 0, len);
                        }
                        pointerRst3.position(0);

                        rst2.addi(rst3);
                    }

                    rst2.permutei(1, 0, 2);
                    return rst2;
                }),

                new OperandInfo(weight, () -> {

                    INDArray rst5 = temp3.expandAndReturnTemp(shape_w);
                    rst5.muli(0);
                    FloatPointer pointerRst5 = (FloatPointer) rst5.data().pointer();
                    INDArray rst6 = temp4.expandAndReturnTemp(N, sxi / groups, L);
                    int len = (int) sxi / groups;
                    float[] floatsRst5 = new float[len];

                    for (int j = 0; j < oc; j++) {

                        Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(
                                input.data.get(NDArrayIndex.all(), NDArrayIndex.interval((j / (oc / groups) * (ic / groups)) * size, ((j / (oc / groups) + 1) * (ic / groups)) * size), NDArrayIndex.all()),
                                grad.reshape(N, oc, L).get(NDArrayIndex.all(), NDArrayIndex.interval(j, j + 1), NDArrayIndex.all()),
                                rst6,
                                0, 2));
                        FloatPointer pointerRst6 = (FloatPointer) rst6.sum(0, 2).data().pointer();
                        pointerRst6.get(floatsRst5, 0, len);
                        pointerRst5.position((j * ic + ic / groups * (j / (oc / groups))) * size);
                        pointerRst5.put(floatsRst5, 0, len);
                    }

                    pointerRst5.position(0);

                  return rst5;
                }),
        };

        Supplier<INDArray> forward = () -> {

            INDArray result = Nd4j.zeros(oc, N, ah, aw);
            FloatPointer pointerResult = (FloatPointer) result.data().pointer();
            FloatPointer pointerRst1;
            int len = (int) N * (int) L;
            float[] floatResult = new float[len];

            for (int i = 0; i < oc; i++) {
                INDArray rst1 = temp1.expandAndReturnTemp(shape_i);
                Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(
                        input.data,
                        weight.data.get(NDArrayIndex.all(), NDArrayIndex.interval(i * HxW * ic, (i + 1) * HxW * ic), NDArrayIndex.all()),
                        rst1,
                        1));
                pointerRst1 = (FloatPointer) rst1.sum(1).data().pointer();
                pointerRst1.get(floatResult, 0, len);
                pointerResult.position(i * len);
                pointerResult.put(floatResult, 0, len);

            }
            pointerResult.position(0);

            result.permutei(1, 0, 2, 3);

            return result;
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);
        setOperatorInfo(operatorInfo);
    }


}
