package LibDL;

import LibDL.TensorImpl.Nd4jTensor;
import org.jetbrains.annotations.NotNull;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.factory.Nd4j;

public class ld {
    private static @NotNull SameDiff sd = SameDiff.create();

    public static @NotNull Tensor tensor(@NotNull int[] values) {
        return new Nd4jTensor(sd.var(Nd4j.create(values)));
    }

    public static @NotNull Tensor ones(@NotNull int[] size) {
        return new Nd4jTensor(sd.var(Nd4j.ones(size)));
    }

    public static @NotNull Tensor randn(@NotNull int[] size) {
        return new Nd4jTensor(sd.var(Nd4j.randn(size)));
    }

    public static @NotNull Tensor randn(int size) {
        return new Nd4jTensor(sd.var(Nd4j.randn(size)));
    }

    public static @NotNull Tensor addmm(@NotNull Tensor mat, @NotNull Tensor mat1, @NotNull Tensor mat2) {
        if (mat instanceof Nd4jTensor && mat1 instanceof Nd4jTensor && mat2 instanceof Nd4jTensor)
            return mat1.mm(mat2).add(mat);
        throw new IllegalArgumentException();
    }

}
