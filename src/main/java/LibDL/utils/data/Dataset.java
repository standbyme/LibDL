package LibDL.utils.data;

import java.util.*;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.stream.Stream;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.omg.CORBA.DATA_CONVERSION;

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

    public ConcatDataset add(Dataset other) {
        return new ConcatDataset(this, other);
    }

    protected abstract Iterator getIteratorByIndex(long index);
}
