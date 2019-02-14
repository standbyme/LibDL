package LibDL.outdated.nn.model;

import LibDL.outdated.nn.model.listeners.Listener;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public interface Model {

    void fit(DataSetIterator iterator);

    INDArray predict(INDArray input);

//    void setOptimizer(Optimizer optimizer);
//
//    Optimizer getOptimizer();

    void addListener(Listener listener);

    default void addListener(Listener... listeners) {
        for (Listener l : listeners) {
            addListener(l);
        }
    }

    void doForward();

    void doBackward();

    void update();

    void setInput(INDArray input);

    INDArray getOutput();

    IActivation getLossActivation();

    INDArray getPreOutput();

    void setError(INDArray error);

    void setLearnRate(double learnRate);
}
