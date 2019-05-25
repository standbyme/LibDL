package LibDL.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

public class INDArrayPointer {

    private INDArray value;
    private String msg;
    private boolean quiet;

    public INDArrayPointer() {
        this.value = Nd4j.zeros(1);
        quiet = true;
    }

    public INDArrayPointer(String msg) {
        this.value = Nd4j.zeros(1);
        this.msg = msg;
        quiet = false;
    }

    public INDArray expandAndReturnTemp(long... shape_i) {
        long[] shape_t = this.value.shape();

        if (shape_t.length != shape_i.length) {
            if (!quiet)
                System.out.print(msg + ":\t" + "rank is too small\t" + Arrays.toString(this.value.shape()) + " >>> ");
            this.value = Nd4j.zeros(shape_i); // TODO
            if (!quiet)
                System.out.println(Arrays.toString(this.value.shape()));
            shape_t = this.value.shape();
        }else {
            int n = shape_t.length;
            long[] expension = Arrays.copyOf(shape_t, n);
            for (int i = 0; i < n; i++) {
                if (shape_t[i] < shape_i[i]) {
                    expension[i] = shape_i[i];
                    if (!quiet)
                        System.out.print(msg + ":\t" + "expand along dim " + i + "\t" + Arrays.toString(this.value.shape()) + " >>> ");
                    this.value = Nd4j.zeros(expension);
                    if (!quiet)
                        System.out.println(Arrays.toString(this.value.shape()));
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
