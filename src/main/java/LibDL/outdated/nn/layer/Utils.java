package LibDL.outdated.nn.layer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class Utils {
    public static INDArray subArray(INDArray input, int x, int y, long width, long height) {
        return input.get(NDArrayIndex.interval(x, x + height), NDArrayIndex.interval(y, y + width));
    }

    public static INDArray getZ(INDArray input, int index) {
        return input.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(index, index + 1));
    }

    public static void putZ(INDArray input, int index, INDArray data) {
        input.put(new INDArrayIndex[]{
                NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(index, index + 1)
        }, data);
    }
}
