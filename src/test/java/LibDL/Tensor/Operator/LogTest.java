package LibDL.Tensor.Operator;

import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class LogTest {
    @Test
    public void testLog() {
        Variable data_to_backward;
        Softmax softmax;
        INDArray target;
        data_to_backward = new Variable(Nd4j.create(new double[][][]{
                {{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}},
                {{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}}
        }), true);
        softmax = new Softmax(data_to_backward, 2);
        Log log = new Log(softmax);
        log.grad = Nd4j.create(new double[][][]{
                {
                        {-7.21521186828613281250, -6.81521177291870117188, -6.81521177291870117188}, {-6.81521177291870117188, -7.81521177291870117188, -6.81521177291870117188}
                },
                {
                        {-6.81521177291870117188, -6.81521177291870117188, -8.41521167755126953125}, {-6.81521177291870117188, -8.01521110534667968750, -6.81521177291870117188}
                }
        });
        target = Nd4j.create(new double[][][]{
                {
                        {-2.40760588645935058594, -1.40760588645935058594, -0.40760588645935058594},
                        {-2.40760588645935058594, -1.40760588645935058594, -0.40760588645935058594}
                },
                {
                        {-2.40760588645935058594, -1.40760588645935058594, -0.40760588645935058594},
                        {-2.40760588645935058594, -1.40760588645935058594, -0.40760588645935058594}
                }
        });
        assertEquals(target, log.data);
        log.backward();
        target = Nd4j.create(new double[][][]{
                {
                        {-5.33846712112426757812, -1.71369075775146484375, 7.05216073989868164062},
                        {-4.88444900512695312500, -2.56685400009155273438, 7.45130395889282226562}
                },
                {
                        {-4.83043050765991210938, -1.42001676559448242188, 6.25044918060302734375},
                        {-4.86644268035888671875, -2.71790742874145507812, 7.58435297012329101562}
                }
        });
        assertEquals(target, data_to_backward.grad);
    }
}
