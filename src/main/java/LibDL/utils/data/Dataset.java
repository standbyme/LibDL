package LibDL.utils.data;

import java.util.Iterator;
import java.util.RandomAccess;
import java.util.Spliterator;
import java.util.function.Consumer;

public abstract class Dataset implements RandomAccess, Iterable {

    @Override
    public Spliterator spliterator() {
        throw new UnsupportedOperationException();
    }

    public abstract long size();

    @Override
    public void forEach(Consumer action) {
        Iterator iterator = iterator();
        while (iterator.hasNext()) {
            action.accept(iterator.next());
        }

    }

    private long batch_size;

    public Dataset() {
        batch_size = -1;
    }

    public void setBatchSize(long batch_size) {
        this.batch_size = batch_size;
    }

    public ConcatDataset add(Dataset other) {
        return new ConcatDataset(this, other);
    }

    protected abstract Iterator getIteratorByIndex(long index);

    public Dataset shuffleData() {
        throw new UnsupportedOperationException();
    }

    public Dataset dropLast(int batch_size) {
        throw new UnsupportedOperationException();
    }
}
