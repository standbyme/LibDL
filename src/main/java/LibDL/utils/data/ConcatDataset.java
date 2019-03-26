package LibDL.utils.data;


import java.util.*;

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

    ConcatDataset(Dataset... datasets) {
        this(Arrays.asList(datasets));
    }

    @Override
    public long size() {
        return cumulative_sizes.lastElement();
    }

    private static long[] getTrueIdx(ConcatDataset dataset, long index) {
        if (index < 0) {
            assert (-index > dataset.size()) :
                    "absolute value of index should" +
                            " not exceed dataset length";
            index = dataset.size() + index;
        }
        long dataset_idx = Collections.binarySearch(dataset.cumulative_sizes, index);
        if (dataset_idx < 0) dataset_idx = -dataset_idx - 1;
        long sample_idx;
        if (dataset_idx == 0) sample_idx = index;
        else {
            sample_idx = index -
                    dataset.cumulative_sizes.get((int) dataset_idx - 1);
        }
        return new long[]{dataset_idx, sample_idx};
    }


    @Override
    protected Iterator getIteratorByIndex(long index) {
        return new ConcatDatasetIter(this, getTrueIdx(this, index));
    }

    private class ConcatDatasetIter implements Iterator {

        private int dataset_idx;
        private long sample_idx;
        private ConcatDataset datasets;
        private Iterator sample_iter;

        ConcatDatasetIter(ConcatDataset datasets, long dataset_idx, long sample_idx) {

            this.dataset_idx = (int) dataset_idx;
            this.sample_idx = sample_idx;

            this.datasets = datasets;
            sample_iter = datasets.datasets.get(this.dataset_idx)
                    .getIteratorByIndex(sample_idx);

        }

        ConcatDatasetIter(ConcatDataset datasets) {
            this(datasets, 0, 0);
        }

        ConcatDatasetIter(ConcatDataset datasets, long[] idxs) {
            this(datasets, idxs[0], idxs[1]);
        }

        @Override
        public boolean hasNext() {
            return sample_idx +
                    datasets.cumulative_sizes.get(dataset_idx)
                    < datasets.cumulative_sizes.lastElement();
        }

        @Override
        public Object next() {
            sample_idx++;
            if (!sample_iter.hasNext()) {
                sample_iter = datasets.datasets.get(++dataset_idx).iterator();
                sample_idx = 0;
            }
            return sample_iter.next();
        }
    }

    @Override
    public Iterator iterator() {
        return new ConcatDatasetIter(this);
    }

}
