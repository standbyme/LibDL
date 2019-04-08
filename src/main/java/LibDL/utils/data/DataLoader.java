package LibDL.utils.data;


import LibDL.utils.Pair;

import java.util.Iterator;

public class DataLoader implements Iterable<Pair> {

    Dataset dataset;
    int batch_size;


    public DataLoader(Dataset dataset,
                      int batch_size,
                      boolean shuffle,
                      boolean drop_last) {

        if (shuffle) {
            this.dataset = dataset.shuffleData();
        } else this.dataset = dataset;
        if (batch_size != -1) {
            this.dataset.batchSize(batch_size);
            if (drop_last) {
                this.dataset = this.dataset.dropLast(true);
            }
        } else this.dataset.batchSize(this.dataset.size());

    }

    @Override
    public Iterator<Pair> iterator() {
        return this.dataset.iterator();
    }
}
