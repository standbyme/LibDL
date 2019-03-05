package LibDL.nn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class WeightInit {

    public static INDArray constant(INDArray paramView, double value) {
        return paramView.assign(value);
    }

    public static INDArray uniform(INDArray paramView, double a, double b) {
        return Nd4j.rand(paramView, Nd4j.getDistributions().createUniform(a, b));
    }

    public static INDArray normal(INDArray paramView, double mean, double std) {
        return Nd4j.randn(paramView).muli(std).addi(mean);
    }

    public static INDArray kaimingUniform(INDArray paramView, double a) {
        double gain = Math.sqrt(2.0 / (1 + a * a));
        double std = gain / Math.sqrt(paramView.size(0));
        double bound = Math.sqrt(3) * std;
        return Nd4j.rand(paramView, Nd4j.getDistributions().createUniform(-bound, bound));
    }
}
