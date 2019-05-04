package LibDL.nn;

import LibDL.Tensor.Operator.Exp;
import LibDL.Tensor.Operator.Log;
import LibDL.Tensor.Operator.Tanh;
import LibDL.Tensor.Tensor;

public class Functional {
    public static Tensor cross_entropy(Tensor tensor, Tensor target) {
        return new SoftmaxCrossEntropyLoss(target).eval(tensor);
    }

    public static Tensor mse_loss(Tensor tensor, Tensor target) {
        return new MSELoss(target).eval(tensor);
    }


    public static Tensor exp(Tensor tensor) {
        return new Exp(tensor);
    }

    public static Tensor log(Tensor tensor) {
        return new Log(tensor);
    }

    public static Tensor tanh(Tensor tensor) {
        return new Tanh(tensor);
    }

    public static Tensor relu(Tensor tensor) {
        return new LibDL.Tensor.Operator.ReLU(tensor);
    }
    public static Tensor dropout(Tensor input, double p, boolean train) {
        return Dropout.dropout_impl(input, p, train, false, false);
    }

    public static Tensor dropout(Tensor input, double p) {
        return dropout(input, p, true);
    }

}
