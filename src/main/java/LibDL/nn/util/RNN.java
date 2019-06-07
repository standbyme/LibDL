package LibDL.nn.util;


import LibDL.Tensor.Tensor;

import java.util.Comparator;
import java.util.stream.IntStream;

public class RNN {
    /*
     input size should be [T, B, ...]
     */
    PackedSequence pack_padded_sequence(Tensor tensor, long[] lengths,
                                        boolean enforce_sorted,
                                        boolean batch_first) {
        if (enforce_sorted) {
            return new PackedSequence(tensor,lengths,null);

        } else {
            long[] sortedIndices = IntStream.range(0, lengths.length)
                    .boxed().sorted(Comparator.comparingLong(i->lengths[i]))
                    .mapToLong(ele -> ele).toArray();
            tensor= tensor.index_select(batch_first?0:1, sortedIndices);

            return new PackedSequence(tensor,lengths,sortedIndices);
        }
    }
}
