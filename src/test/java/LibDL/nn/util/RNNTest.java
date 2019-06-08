package LibDL.nn.util;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Tensor;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

public class RNNTest {

    @Test
    public void pack_padded_sequence() {
        Tensor value = new Constant(
                Nd4j.create(new double[][]{
                        {1, 2, 3, 4, 5, 6, 7, 0, 0, 0},
                        {1, 2, 3, 4, 5, 6, 0, 0, 0, 0},
                        {1, 2, 3, 4, 5, 0, 0, 0, 0, 0},
                        {1, 2, 3, 4, 5, 6, 7, 8, 0, 0},
                        {1, 2, 3, 4, 5, 6, 7, 0, 0, 0},
                })
        );
        PackedSequence packedSequence = RNN.pack_padded_sequence(value, new long[]{
                7, 6, 5, 8, 7
        }, false, true);

        System.out.println(packedSequence);
    }

    @Test
    public void pad_packed_sequence() {
    }
}