package LibDL.utils.data;

import org.jetbrains.annotations.NotNull;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;
import java.util.function.Consumer;

public class ConcatDataset extends Dataset {

    private List<Dataset> datasets;
    private Vector<Long> cumulative_sizes;

    private static Vector<Long> cumsum(List<Dataset> seq) {
        Vector<Long> r = new Vector<>();
        long s = 0;
        for (Dataset dataset : seq) {
            long l = dataset.size();
            assert l > 0 : "datasets should not be an empty iterable";
            r.add(l + s);
            s += l;
        }
        return r;
    }

    ConcatDataset(List<Dataset> datasets) {
        assert datasets.size() > 0 : "datasets should not be an empty iterable";
        this.datasets = datasets;
        this.cumulative_sizes = cumsum(datasets);

    }

    @Override
    public long size() {
        return cumulative_sizes.lastElement();
    }

//    private static Pair<Long, Long> getItemByIndex(ConcatDataset dataset, long index) {
//        if (index < 0) {
//            assert (-index > dataset.size()) :
//                    "absolute value of index should" +
//                            " not exceed dataset length";
//            index = dataset.size() + index;
//        }
//        long dataset_idx = Collections.binarySearch(dataset.cumulative_sizes, index);
//        if (dataset_idx < 0) dataset_idx = -dataset_idx - 1;
//        long sample_idx;
//        if (dataset_idx == 0) sample_idx = index;
//        else {
//            sample_idx = index -
//                    dataset.cumulative_sizes.get((int) dataset_idx - 1);
//        }
//        return new Pair<Long, Long>(dataset_idx, sample_idx);
//    }


    private class ConcatDatasetIter implements Iterator {

        private int dataset_idx;
        private long sample_idx;
        private ConcatDataset datasets;
        private Iterator sample_iter;

        ConcatDatasetIter(ConcatDataset datasets) {
            sample_idx = dataset_idx = 0;
            this.datasets = datasets;
            sample_iter = datasets.datasets.get(0).iterator();

        }

        @Override
        public boolean hasNext() {
            return sample_idx < datasets.cumulative_sizes.lastElement();
        }

        @Override
        public Object next() {
            sample_idx++;
            if (!sample_iter.hasNext())
                sample_iter = datasets.datasets.get(++dataset_idx).iterator();
            return sample_iter.next();
        }
    }

    @Override
    public Iterator iterator() {
        return new ConcatDatasetIter(this);
    }

}
