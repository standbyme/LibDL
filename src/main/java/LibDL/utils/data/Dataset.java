package LibDL.utils.data;

import java.util.*;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.stream.Stream;

import javafx.util.Pair;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public abstract class Dataset implements RandomAccess, Iterable {

    @Override
    public Spliterator spliterator() {
        throw new UnsupportedOperationException();
    }

    public abstract long size();
}
