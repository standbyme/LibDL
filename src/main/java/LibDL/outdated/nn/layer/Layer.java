package LibDL.outdated.nn.layer;


import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface Layer {

    IActivation getActivationFunction();

    void setActivationFunction(IActivation activationFunction);

    String getName();

    void setName(String name);

    void doForward();

    void doBackward();

    void update();

    void setInput(INDArray input);

    INDArray getOutput();

    void setError(INDArray output);

    INDArray getEpsilon();

    void setLearnRate(double learnRate);

    INDArray run(INDArray input);
}
