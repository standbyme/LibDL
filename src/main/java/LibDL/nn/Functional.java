package LibDL.nn;

import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Variable;
import LibDL.Tensor.Tensor;

public class Functional {
    public static CrossEntropyLoss cross_entropy(Tensor tensor, Tensor target) {
        CrossEntropyLoss loss = new CrossEntropyLoss(target);
        loss.setInput(tensor);
        loss.forwardThisLayer();
        return loss;
    }

    public static MSELoss mse_loss(Tensor tensor, Variable target) {
        MSELoss loss = new MSELoss(target);
        loss.setInput(tensor);
        loss.forwardThisLayer();
        return loss;
    }

//    public static Tensor linear(Tensor input){
//        Dense layer= new Dense()
//    }

}
