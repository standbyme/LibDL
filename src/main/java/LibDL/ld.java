package LibDL;

import LibDL.TensorImpl.Nd4jTensor;
import org.jetbrains.annotations.NotNull;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.factory.Nd4j;

public class ld {
    static SameDiff sd = SameDiff.create();

    public static @NotNull Nd4jTensor tensor(@NotNull int[] values) {
        return new Nd4jTensor(sd.var(Nd4j.create(values)));
    }

    public static @NotNull Nd4jTensor randn(@NotNull int[] size) {
        return new Nd4jTensor(sd.var(Nd4j.randn(size)));
    }

    public static @NotNull Nd4jTensor randn(int size) {
        return new Nd4jTensor(sd.var(Nd4j.randn(size)));
    }

    public static @NotNull Nd4jTensor addmm(@NotNull Tensor b, @NotNull Tensor input, @NotNull Tensor w) {
        if (b instanceof Nd4jTensor && input instanceof Nd4jTensor && w instanceof Nd4jTensor)
            return new Nd4jTensor(sd.var(Nd4j.empty()));
        throw new IllegalArgumentException();
    }

}
