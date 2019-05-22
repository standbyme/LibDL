package LibDL.NumImpl;

import LibDL.Num;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Nd4jNum implements Num {

    private INDArray data;

    private Nd4jNum(INDArray data) {
        this.data = data;
    }

    public static Nd4jNum randn(@NotNull int[] size) {
        return new Nd4jNum(Nd4j.randn(size));
    }

    public static Nd4jNum randn(int size) {
        return new Nd4jNum(Nd4j.randn(size));
    }

    public static Nd4jNum empty() {
        return new Nd4jNum(Nd4j.empty());
    }

}
