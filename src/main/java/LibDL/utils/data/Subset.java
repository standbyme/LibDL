package LibDL.utils.data;

import java.util.Iterator;

public class Subset extends Dataset {


    @Override
    protected Iterator getIteratorByIndex(long index) {
        return null;
    }

    @Override
    public long size() {
        return 0;
    }

    @Override
    public Iterator iterator() {
        return null;
    }
}
