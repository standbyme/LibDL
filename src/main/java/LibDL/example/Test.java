package LibDL.example;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Test {
    public static void main(String[] args) {
        INDArray x = Nd4j.randn(2, 3);
        INDArray y = Nd4j.randn(3, 4);

        INDArray z = x.mmul(y);
        System.out.println(z);
        System.out.println(z.mul(2));

        INDArray x2 = x.reshape(2,3);
        System.out.println(x2);
        System.out.println(x2.mul(2));


        INDArray z2 = x.mmul(y).reshape(2, 4);
        System.out.println(z2);
        System.out.println(z2.mul(2));

    }
}
