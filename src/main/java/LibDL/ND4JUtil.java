package LibDL;

import org.jetbrains.annotations.Nullable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class ND4JUtil {
    private static INDArray exec(TransformOp op) {
        if (op.x().isCleanedUp()) throw new IllegalStateException("NDArray already freed");
        return Nd4j.getExecutioner().execAndReturn(op);
    }

    public static INDArray pow(INDArray x, int exponent) {
        return exec(new org.nd4j.linalg.api.ops.impl.transforms.Pow(x, x.dup(), exponent));
    }

    public static INDArray Exp(INDArray x) {
        return exec(new org.nd4j.linalg.api.ops.impl.transforms.Exp(x.dup()));
    }

    public static INDArray ReLU(INDArray x) {
        return exec(new org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear(x.dup()));
    }

    public static INDArray Log(INDArray x) {
        return exec(new org.nd4j.linalg.api.ops.impl.transforms.Log(x.dup()));
    }

    public static INDArray Step(INDArray x) {
        return exec(new org.nd4j.linalg.api.ops.impl.transforms.Step(x.dup()));
    }

    public static INDArray TanhDerivative(INDArray x) {
        return exec(new org.nd4j.linalg.api.ops.impl.transforms.TanhDerivative(x.dup()));
    }


    private static INDArrayIndex[] construct_indices_array_impl(int rank, int dim, INDArrayIndex index) {
        INDArrayIndex[] indArrayIndices = new INDArrayIndex[rank];
        for (int i = 0; i < rank; i++) {
            if (i != dim) {
                indArrayIndices[i] = NDArrayIndex.all();
            } else {
                indArrayIndices[i] = index;
            }
        }
        return indArrayIndices;
    }

    public static INDArrayIndex[] construct_indices_array(int rank, int dim) {
        return construct_indices_array_impl(rank, dim, null);
    }

    public static INDArrayIndex[] construct_indices_array(int rank, int dim, long... index) {
        return construct_indices_array(rank, dim, NDArrayIndex.indices(index));
    }

    public static INDArrayIndex[] construct_indices_array(int rank, int dim, @Nullable INDArrayIndex index) {
        return construct_indices_array_impl(rank, dim, index);
    }

    public static INDArrayIndex[] construct_chop_indices_array(int rank, int dim, long begin, long end) {
        return construct_indices_array(rank, dim, NDArrayIndex.interval(begin,end));
    }

}
