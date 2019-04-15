package LibDL.nn;

import LibDL.Tensor.Variable;
import LibDL.Tensor.Tensor;

public class Functional {
    public static SoftmaxCrossEntropyLoss cross_entropy(Tensor tensor, Tensor target) {
        SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss(target);
        loss.apply(tensor);
        return loss;
    }

    public static MSELoss mse_loss(Tensor tensor, Variable target) {
        MSELoss loss = new MSELoss(target);
        loss.apply(tensor);
        return loss;
    }

//    public static Tensor linear(Tensor input){
//        Dense layer= new Dense()
//    }

}
