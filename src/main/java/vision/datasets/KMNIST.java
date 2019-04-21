package vision.datasets;

public class KMNIST extends MNIST {
    public KMNIST(String root, boolean train) {
        super(root, train);
    }

    public KMNIST(String root, boolean train, boolean one_hot) {
        super(root, train, one_hot);
    }
}
