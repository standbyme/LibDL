package LibDL.nn.util;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Tensor;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
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
        }, false, false);

        System.out.println(packedSequence);

        Tensor tensor = RNN.pad_packed_sequence(packedSequence,packedSequence);
        System.out.println(tensor);
    }

    @Test
    public void pad_packed_sequence() {
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
        }, false, false);
        PackedSequence packedSequence1 = new PackedSequence(1);
        for (int i = 0; i < packedSequence.size(); i++) {
            INDArray now = packedSequence.get(i);
            System.out.println(now);
            packedSequence1.add(now);
        }
        packedSequence1.form();
        Tensor reversed = RNN.pad_packed_sequence(packedSequence1,packedSequence);

        System.out.println(reversed);
    }
}