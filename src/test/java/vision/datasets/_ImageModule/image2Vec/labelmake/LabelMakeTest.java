package vision.datasets._ImageModule.image2Vec.labelmake;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public class LabelMakeTest {

    private List<String> features = Arrays.asList("1", "2", "3");

    private Function<String, INDArray> function = new Function<String, INDArray>() {
        @Override
        public INDArray apply(String s) {
            int s_int = Integer.parseInt(s);
            INDArray res = Nd4j.ones(4);
            res.putScalar(s_int, 0);
            return res;
        }
    };

    @Test
    public void labelMakeTest() {
        System.out.println(LabelMake.labelMake(features, function, 3, 4));
    }
}
