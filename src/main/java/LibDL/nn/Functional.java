package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Tensor;

public class Functional {
    public static CrossEntropyLoss cross_entropy(Tensor tensor, Tensor target) {
        CrossEntropyLoss loss = new CrossEntropyLoss(target);
        loss.setInput(tensor);
        return loss;
    }

    public static MSELoss mse_loss(Tensor tensor, Constant target) {
        MSELoss loss = new MSELoss(target);
        loss.setInput(tensor);
        return loss;
    }

}
