package vision.datasets;

import LibDL.utils.data.Dataset;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.factory.Nd4j;

public class FakeData extends Dataset {
    private static int[] combineShape(int size, int[] image_size) {
        return ArrayUtils.insert(0, image_size, size);
    }

    public FakeData(int size, int[] image_size, int num_classes) {
        super(Nd4j.rand(combineShape(size, image_size)),
                Nd4j.rand(new int[]{size, num_classes}));
    }
}
