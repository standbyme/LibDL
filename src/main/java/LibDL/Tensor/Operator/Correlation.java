package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.bytedeco.javacpp.FloatPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.function.Supplier;

public class Correlation extends OperatorTensor {

    static class INDArrayPointer {

        INDArray value;

        INDArrayPointer() {
            this.value = Nd4j.zeros(1);
        }

        INDArray expandAndReturnTemp(long[] shape_i) {
            long[] shape_t = this.value.shape();

            if (shape_t.length != shape_i.length) {
                System.out.print(Arrays.toString(this.value.shape()) + " >>> ");
                this.value = Nd4j.zeros(shape_i); // TODO
                System.out.println("new:" + "temp:shape:" + Arrays.toString(this.value.shape()));
                shape_t = this.value.shape();
            }else {
                int n = shape_t.length;
                long[] expension = Arrays.copyOf(shape_t, n);
                for (int i = 0; i < n; i++) {
                    if (shape_t[i] < shape_i[i]) {
                        expension[i] = shape_i[i];
                        System.out.print(Arrays.toString(this.value.shape()) + " >>> ");
                        this.value = Nd4j.zeros(expension);
                        System.out.println("new:" + i + "temp:shape:" + Arrays.toString(this.value.shape()));
                        shape_t = this.value.shape();
                    }
                }
            }
            INDArrayIndex[] indArrayIndices = new INDArrayIndex[shape_t.length];
            for (int i = 0; i < shape_t.length; i++) {
                indArrayIndices[i] = NDArrayIndex.interval(0, 1, shape_i[i]);
            }
            return this.value.get(indArrayIndices);
        }
    }

    private static INDArrayPointer temp0 = new INDArrayPointer();
    private static INDArrayPointer temp1 = new INDArrayPointer();
    private static INDArrayPointer temp2 = new INDArrayPointer();
    private INDArray test; // TODO delete

    Correlation(Tensor input, Tensor weight, long ah, long aw, int ic, int oc, int groups) {

        assert input.data.rank() == 3;
        assert weight.data.rank() == 3;
        assert weight.data.shape()[0] == 1 && weight.data.shape()[2] == 1;

        long[] shape_i = input.data.shape();
        long[] shape_w = weight.data.shape();

        long N = shape_i[0];
        long L = shape_i[2];
        long HxW = shape_w[1] / oc / ic;
        long sxi = shape_i[1]; // size * ic

        OperandInfo[] operandInfos = new OperandInfo[]{
                new OperandInfo(input, () -> {

                    INDArray rst2 = Nd4j.zeros(sxi, N, L);

                    INDArray rst3 = Nd4j.zeros(sxi, N, L);
                    FloatPointer pointerRst3 = (FloatPointer) rst3.data().pointer();

                    INDArray rst4 = temp2.expandAndReturnTemp(new long[]{N, 1, L});
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
                   return null;
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
