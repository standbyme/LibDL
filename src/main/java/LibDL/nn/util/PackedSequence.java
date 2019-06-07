package LibDL.nn.util;

import LibDL.ND4JUtil;
import LibDL.Tensor.Tensor;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;

public class PackedSequence extends Tensor {

    long[] sorted_index;
    long[] batch_sizes;
    ArrayList<long[]> batch_shape;
    ArrayList<long[]> batch_where;
    INDArrayIndex[] indices;
    int batch_dim;
    INDArray data_;
//    ArrayList<Long>

    @Override
    public void backward() {
        throw new UnsupportedOperationException();
    }

    public PackedSequence(@NotNull Tensor data,
                          @NotNull long[] batch_sizes,
                          @Nullable long[] sorted_index,
                          int batch_dim) {
        this.sorted_index = sorted_index;
        this.batch_sizes = batch_sizes;
        this.data = Nd4j.create();
        this.data_ = data.data.dup(new char[]{'c', 'f'}[batch_dim]);
        this.batch_shape = new ArrayList<>();
        this.batch_where = new ArrayList<>();
        this.batch_dim = batch_dim;
        this.indices = ND4JUtil.construct_indices_array(data_.rank(), batch_dim);
//        this.data
        work();
    }

    private void copy_at(long batch_sz, long from, long to) {
        indices[batch_dim] = NDArrayIndex.interval(0, batch_sz);
        indices[1 - batch_dim] = NDArrayIndex.interval(to + 1, from + 1);
        INDArray now_data = this.data_.get(indices);
        this.data = Nd4j.concat(0, this.data,
                Nd4j.toFlattened(now_data));
        batch_where.add(new long[]{to + 1, from + 1});
        batch_shape.add(now_data.shape());

    }

    private void work() {
//        long dim_size = this.data_.size(1 - batch_dim);
        int last_same_batch = batch_sizes.length - 1;
        for (int i = batch_sizes.length - 1; i >= 0; ) {
            if (batch_sizes[i] == batch_sizes[last_same_batch]) i--;
            else {
                copy_at(batch_sizes[i], last_same_batch, i);

                last_same_batch = i;
            }
        }
        if (last_same_batch != 0) {
            copy_at(batch_sizes[0], last_same_batch, -1);
        }
    }

    public INDArray get(int i) {
        return this.data.get(NDArrayIndex.interval(
                batch_where.get(i)[0],
                batch_where.get(i)[1]
        )).reshape(batch_shape.get(i));
    }

    @Override
    public long size(int i) {
        if (i == batch_dim) return batch_sizes[0];//max batch size
        return super.size(i);
    }

    public long size() {
        return this.batch_shape.size();
    }
}
