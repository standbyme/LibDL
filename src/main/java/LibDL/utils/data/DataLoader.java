package LibDL.utils.data;


import LibDL.utils.Pair;

import java.util.Iterator;
import java.util.function.Consumer;

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
        if (drop_last) {
            this.dataset = this.dataset.dropLast(batch_size);
        }

    }

    @Override
    public Iterator<Pair> iterator() {
        return null;
    }
}
