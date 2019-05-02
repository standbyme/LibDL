package LibDL.nn;

import LibDL.Tensor.Tensor;

public class Functional {
    public static Tensor cross_entropy(Tensor tensor, Tensor target) {
        SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss(target);
        return loss.eval(tensor);
    }

    public static Tensor mse_loss(Tensor tensor, Tensor target) {
        MSELoss loss = new MSELoss(target);
        return loss.eval(tensor);
    }

    public static Tensor relu(Tensor tensor) {
        return new LibDL.Tensor.Operator.ReLU(tensor);
    }
}
