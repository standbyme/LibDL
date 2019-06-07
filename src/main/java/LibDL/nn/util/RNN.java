package LibDL.nn.util;


import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;

import java.util.Comparator;
import java.util.stream.IntStream;

public class RNN {
    /*
     input size should be [T, B, ...]
     */
    PackedSequence pack_padded_sequence(Tensor tensor, long[] lengths,
                                        boolean enforce_sorted,
                                        boolean batch_first) {
        int batch_dim = batch_first ? 0 : 1;

        if (enforce_sorted) {
            return new PackedSequence(tensor, lengths, null, batch_dim);
        } else {
            long[] sortedIndices = IntStream.range(0, lengths.length)
                    .boxed().sorted(Comparator.comparingLong(i -> lengths[i]))
                    .mapToLong(ele -> ele).toArray();
            tensor = tensor.index_select(batch_dim, sortedIndices);
            return new PackedSequence(tensor, lengths, sortedIndices, batch_dim);
        }
    }

    Tensor pad_packed_sequence(PackedSequence packedSequence) {
        return new Variable(packedSequence.to_original());
    }
}
