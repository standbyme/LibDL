package LibDL;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.strict.TanhDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class ND4JUtil {
    public static INDArray pow(INDArray x, int exponent) {
        return Transforms.pow(x, exponent);
    }

    public static INDArray Exp(INDArray x) {
        return Transforms.exp(x);
    }

    public static INDArray ReLU(INDArray x) {
        return Transforms.relu(x);
    }

    public static INDArray Log(INDArray x) {
        return Transforms.log(x);
    }

    public static INDArray Step(INDArray x) {
        return Transforms.step(x);
    }

    public static INDArray TanhDerivative(INDArray x) {
        return Nd4j.exec(new TanhDerivative(x, x.ulike()));
    }
}
