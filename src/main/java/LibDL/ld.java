package LibDL;

import LibDL.NumImpl.Nd4jNum;
import org.jetbrains.annotations.NotNull;

public class ld {
    public static Tensor tensor(@NotNull Num value) {
        return new Tensor(value);
    }

    public static Tensor randn(@NotNull int[] size) {
        return new Tensor(Nd4jNum.randn(size));
    }

    public static Tensor randn(int size) {
        return new Tensor(Nd4jNum.randn(size));
    }

    public static Tensor addmm(@NotNull Tensor b, @NotNull Tensor input, @NotNull Tensor w) {
        return new Tensor(Nd4jNum.empty());
    }
}
