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
    long max_batch_size;
    INDArray data_;
    boolean from_none;
//    ArrayList<Long>

    @Override
    public void backward() {
        throw new UnsupportedOperationException();
    }

    private PackedSequence() {
        //universal init
        this.data = Nd4j.empty();
        this.batch_shape = new ArrayList<>();
        this.batch_where = new ArrayList<>();
        this.max_batch_size = 0;
    }

    public PackedSequence(int batch_dim) {
        this();
        this.batch_dim = batch_dim;
        this.from_none = true;
    }

    public PackedSequence(@NotNull Tensor data,
                          @NotNull long[] batch_sizes,
                          @Nullable long[] sorted_index,
                          int batch_dim) {
        this();
        this.max_batch_size = batch_sizes[0];
        this.sorted_index = sorted_index;
        this.batch_sizes = batch_sizes;
        this.data_ = data.data.dup(new char[]{'c', 'f'}[batch_dim]);
        this.batch_dim = batch_dim;
        this.indices = ND4JUtil.construct_indices_array(data_.rank(), batch_dim);
        this.from_none = false;
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
        if (i == batch_dim) return max_batch_size;//max batch size
        return super.size(i);
    }

    public long size() {
        return this.batch_shape.size();
    }

    public void add(INDArray array) {
        long from = this.data.size(0);
        this.data = Nd4j.concat(0, this.data, Nd4j.toFlattened(array));
        this.batch_where.add(new long[]{from, this.data.size(0)});
        this.batch_shape.add(array.shape());
    }

    public INDArray from_packed() {
        INDArray ret = Nd4j.empty();
        for (int i = 0; i < size(); i++) {
            INDArray now = get(i);
            long[] shape = batch_shape.get(i);
            shape[0] = max_batch_size - shape[0];
            ret = Nd4j.concat(1 - batch_dim, ret,
                    Nd4j.concat(0, now, Nd4j.zeros(shape)));
        }
        return ret;
    }

    private long[] reverse_sorted() {
        long[] reversed = new long[sorted_index.length];
        long j = 0;
        for (long i : sorted_index)
            reversed[(int) i] = j++;
        return reversed;
    }

    public INDArray to_original() {
        if (from_none)
            return from_packed();
        if (this.sorted_index != null)
            return this.data_.get(ND4JUtil.construct_indices_array(
                    this.data_.rank(), 1 - this.batch_dim,
                    reverse_sorted()
            ));
        return this.data_;
    }
}
