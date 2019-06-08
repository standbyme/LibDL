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
import java.util.Arrays;

public class PackedSequence extends Tensor {

    long[] sorted_index;
    long[] batch_sizes;
    ArrayList<long[]> batch_shape;
    ArrayList<long[]> batch_where;
    ArrayList<INDArrayIndex[]> batch_put;
    ArrayList<INDArray> arrays;
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
        this.batch_put = new ArrayList<>();
        this.arrays = new ArrayList<>();
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
        System.out.println(this.data_);
//        this.data
        work();
    }

    private void copy_at(long batch_sz, long from, long to, long new_batch_sz) {
        indices[batch_dim] = NDArrayIndex.interval(batch_sz, new_batch_sz);
        indices[1 - batch_dim] = NDArrayIndex.interval(0, from + 1);
        INDArray now_data = this.data_.get(indices);
        System.out.println("Copy at " + batch_sz + ", " + " from " + from + ", to " + to);
        System.out.println(now_data);
        this.arrays.add(Nd4j.toFlattened(now_data));
        this.batch_put.add(this.indices.clone());
        batch_shape.add(now_data.shape());

    }

    private void work() {
        long lst_batch_sz = 0;
//        long dim_size = this.data_.size(1 - batch_dim);
        int last_same_batch = batch_sizes.length - 1;
        for (int i = batch_sizes.length - 1; i >= 0; ) {
            if (batch_sizes[i] == batch_sizes[last_same_batch]) i--;
            else {
                copy_at(lst_batch_sz, last_same_batch, i, batch_sizes[last_same_batch]);
                lst_batch_sz = batch_sizes[last_same_batch];
                last_same_batch = i;
            }
        }
//        System.out.println("Last same batch "+ last_same_batch);
//        if (last_same_batch != 0) {
//            System.out.println("Last one");
        copy_at(lst_batch_sz, last_same_batch, -1, batch_sizes[last_same_batch]);
//        }

        this.form();

        for (long[] array : this.batch_where) {
            System.out.println(Arrays.toString(array));
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
//        long from = this.data.size(0);
        this.arrays.add(Nd4j.toFlattened(array));
//        this.batch_where.add(new long[]{from, this.data.size(0)});
        this.batch_shape.add(array.shape());
        this.max_batch_size += array.size(batch_dim);
    }

    public void form() {
        long n = 0;
        for (INDArray x : arrays) {
            batch_where.add(new long[]{n, n + x.size(1)});
            n += x.size(1);
        }
        this.data = Nd4j.concat(1, arrays.toArray(new INDArray[]{}));
//        System.out.println(Arrays.toString(data.shape()));
    }

    private int[] create_batch_shape(int which) {
        long[] shape = batch_shape.get(which).clone();
        shape[batch_dim] = max_batch_size;
        return Arrays.stream(shape).mapToInt(i -> (int) i).toArray();
    }

    public INDArray from_packed(ArrayList<INDArrayIndex[]> put) {
        INDArray ret = Nd4j.empty();
        int[] shape = create_batch_shape(0);
        INDArray narray = Nd4j.zeros(shape);
        for (int i = 0; i < size(); i++) {
//            INDArray sbarray= this.arrays.get(i);
//            if(sbarray.isScalar())sbarray
            narray.put(put.get(i), this.arrays.get(i).reshape(batch_shape.get(i)));
        }
        return narray;
    }

    private long[] reverse_sorted() {
        long[] reversed = new long[sorted_index.length];
        long j = 0;
        for (long i : sorted_index)
            reversed[(int) i] = j++;
        System.out.println(Arrays.toString(reversed));
        return reversed;
    }

    public INDArray to_original(PackedSequence sequence) {
        if (from_none)
            return from_packed(sequence.batch_put);
        if (this.sorted_index != null)
            return this.data_.get(ND4JUtil.construct_indices_array(
                    this.data_.rank(), 1 - this.batch_dim,
                    reverse_sorted()
            ));
        return this.data_;
    }
}
