package LibDL.nn;

import LibDL.Tensor.Operator.ReLU;
import LibDL.Tensor.Tensor;

public class Functional {
    public static Tensor cross_entropy(Tensor tensor, Tensor target) {
        return new SoftmaxCrossEntropyLoss(target).eval(tensor);
    }

    public static Tensor mse_loss(Tensor tensor, Tensor target) {
        return new MSELoss(target).eval(tensor);
    }


    public static Tensor dropout(Tensor input, double p, boolean train) {
        return Dropout.dropout_impl(input, p, train, false, false);
    }

    public static Tensor dropout(Tensor input, double p) {
        return dropout(input, p, true);
    }

    public static Tensor sigmoid(Tensor input) {
        return Tensor.ones(input.sizes()).div(Tensor.exp(input.mul(-1)).add(1));
    }

    public static Tensor relu(Tensor input) {
        return new ReLU(input);
    }

}
