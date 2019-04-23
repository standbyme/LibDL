package vision.datasets;

public class FashionMNIST extends MNIST {
    public FashionMNIST(String root, boolean train) {
        super(root, train);
    }

    public FashionMNIST(String root, boolean train, boolean one_hot) {
        super(root, train, one_hot);
    }
}
