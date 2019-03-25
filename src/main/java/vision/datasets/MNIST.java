package vision.datasets;

import LibDL.utils.data.Dataset;
import com.jstarcraft.module.idxUbyte2Vec.IdxUbyteRead;
import javafx.util.Pair;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Iterator;
import java.util.function.Consumer;

public class MNIST extends Dataset {
    private static String[] test_name = {
            "t10k-images.idx3-ubyte",
            "t10k-labels.idx1-ubyte",
    };
    private static String[] train_name = {
            "train-images.idx3-ubyte",
            "train-labels.idx1-ubyte"
    };
    public INDArray train_data, train_label, test_data, test_label;

    private boolean isTrain;

    public MNIST(String root, boolean train) {
        isTrain = train;
        if (train) {
            train_data = IdxUbyteRead.fromFile(root + train_name[0]);
            train_label = IdxUbyteRead.fromFile(root + train_name[1]);
            assert train_data.size(0) == train_label.size(0);
        } else {
            test_data = IdxUbyteRead.fromFile(root + test_name[0]);
            test_label = IdxUbyteRead.fromFile(root + test_name[1]);
            assert test_data.size(0) == test_label.size(0);
        }
    }

    private class MNISTIterator implements Iterator<Pair<INDArray, INDArray>> {

        private INDArray data, label;
        private int now;

        public MNISTIterator(INDArray data, INDArray label, int now) {
            this.data = data;
            this.label = label;
            this.now = now;
        }

        @Override
        public boolean hasNext() {
            return now < data.size(0);
        }

        @Override
        public Pair<INDArray, INDArray> next() {
            Pair<INDArray, INDArray> ret = new Pair<>(data.getRow(now), label.getRow(now));
            now++;
            return ret;
        }
    }

    @Override
    public long size() {
        if (isTrain)
            return train_data.size(0);
        else
            return test_data.size(0);
    }

    @Override
    public void forEach(Consumer action) {
        Iterator iterator = iterator();
        while (iterator.hasNext()) {
            action.accept(iterator.next());
        }

    }

    @NotNull
    @Override
    public Iterator iterator() {
        if (isTrain) {
            return new MNISTIterator(train_data, train_label, 0);
        } else {
            return new MNISTIterator(test_data, test_label, 0);
        }
    }

}
