package vision.datasets;

import LibDL.utils.data.Dataset;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import vision.datasets._ImageModule.idxUbyte2Vec.IdxUbyteRead;

import java.util.Iterator;

public class MNIST extends Dataset {
    private static String[] test_name = {
            "t10k-images.idx3-ubyte",
            "t10k-labels.idx1-ubyte",
    };
    private static String[] train_name = {
            "train-images.idx3-ubyte",
            "train-labels.idx1-ubyte"
    };
    private boolean isTrain;


    private INDArray createOneHot(INDArray array, int sz) {
        INDArray one_hot = Nd4j.zeros(array.size(0) * sz, 1);
        INDArray ones = Nd4j.onesLike(array);
        INDArray indices = Nd4j.linspace(0, array.size(0) - 1,
                array.size(0)).transpose();
        indices = indices.mul(sz).add(array);
        one_hot.put(new INDArrayIndex[]{NDArrayIndex.indices(indices.data().asLong()), NDArrayIndex.all()}, ones);
        return one_hot.reshape(array.size(0), sz);
    }

    public MNIST(String root, boolean train) {
        this(root, train, false);
    }

    public MNIST(String root, boolean train, boolean one_hot) {
        isTrain = train;
        if (train) {
            data = IdxUbyteRead.fromFile(root + train_name[0]);
            target = IdxUbyteRead.fromFile(root + train_name[1]);
        } else {
            data = IdxUbyteRead.fromFile(root + test_name[0]);
            target = IdxUbyteRead.fromFile(root + test_name[1]);
        }

        if (one_hot) target = createOneHot(target, 10);

        assert data.size(0) == target.size(0);
    }


}
