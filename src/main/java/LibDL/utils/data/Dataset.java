package LibDL.utils.data;

import LibDL.utils.Pair;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.Iterator;
import java.util.RandomAccess;
import java.util.Spliterator;
import java.util.function.Consumer;

public class Dataset implements RandomAccess, Iterable {

    @Override
    public Spliterator spliterator() {
        throw new UnsupportedOperationException();
    }


    @Override
    public void forEach(Consumer action) {
        Iterator iterator = iterator();
        while (iterator.hasNext()) {
            action.accept(iterator.next());
        }

    }

    public INDArray data, target;
    private boolean drop_last;

    public Dataset reshapeData(long... newShape) {

        newShape = ArrayUtils.insert(0, newShape, this.size());
        System.out.println(Arrays.toString(newShape));

        data = data.reshape(newShape);
        return this;
    }

    public Dataset reshapeTarget(long... newShape) {
        newShape = ArrayUtils.insert(0, newShape, this.size());
        target.reshape(newShape);
        return this;
    }

    protected Dataset(INDArray data, INDArray target) {
        this();
        this.data = data;
        this.target = target;
    }

    private class DatasetIterator implements Iterator<Pair<INDArray, INDArray>> {

        private Dataset dataset;
        private long now;

        public DatasetIterator(Dataset dataset, long now) {
            this.dataset = dataset;
            this.now = now;
        }


        @Override
        public boolean hasNext() {
            if (dataset.drop_last) {
                return (dataset.data.size(0) - now * dataset.batch_size) >= dataset.batch_size;
            } else return now * dataset.batch_size < dataset.data.size(0);
        }

        @Override
        public Pair<INDArray, INDArray> next() {
            INDArrayIndex[] dataIndex = new INDArrayIndex[dataset.data.rank()];
            INDArrayIndex[] targetIndex = new INDArrayIndex[dataset.target.rank()];
            INDArrayIndex first = NDArrayIndex.interval(now * dataset.batch_size,
                    Math.min((now + 1) * dataset.batch_size, dataset.size()));

            dataIndex[0] = targetIndex[0] = first;
            for (int i = 1; i < dataIndex.length; i++) dataIndex[i] = NDArrayIndex.all();
            for (int i = 1; i < targetIndex.length; i++) targetIndex[i] = NDArrayIndex.all();

            Pair<INDArray, INDArray> ret = new Pair(dataset.data.get(dataIndex),
                    dataset.target.get(targetIndex));
            now++;
            return ret;
        }
    }

    public long size() {
        return data.size(0);
    }

    @Override
    public Iterator iterator() {
        return new DatasetIterator(this, 0);
    }

    private long batch_size;

    public Dataset() {
        batch_size = 1;
    }

    public Dataset batchSize(long batch_size) {
        this.batch_size = batch_size;
        return this;
    }

    public Dataset add(Dataset other) {
        return new Dataset(Nd4j.concat(0, data, other.data),
                Nd4j.concat(0, data, other.data));
    }

//    protected abstract Iterator getIteratorByIndex(long index);

    public Dataset shuffleData() {
        throw new UnsupportedOperationException();
    }

    public Dataset dropLast(boolean drop_last) {
        this.drop_last = drop_last;
        return this;
    }
}
