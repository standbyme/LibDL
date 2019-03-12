package LibDL;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.TransformOp;
import org.nd4j.linalg.factory.Nd4j;

public class ND4JUtil {
    private static INDArray exec(TransformOp op) {
        if (op.x().isCleanedUp()) throw new IllegalStateException("NDArray already freed");
        return Nd4j.getExecutioner().execAndReturn(op);
    }

    public static INDArray pow(INDArray x, int exponent) {
        return exec(new org.nd4j.linalg.api.ops.impl.transforms.Pow(x, x.dup(), exponent));
    }

    public static INDArray Exp(INDArray x){
        return exec(new org.nd4j.linalg.api.ops.impl.transforms.Exp(x.dup()));
    }

    public static INDArray ReLU(INDArray x){
        return exec(new org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear(x.dup()));
    }

    public static INDArray Log(INDArray x){
        return exec(new org.nd4j.linalg.api.ops.impl.transforms.Log(x.dup()));
    }

    public static INDArray Step(INDArray x){
        return exec(new org.nd4j.linalg.api.ops.impl.transforms.Step(x.dup()));
    }
}
