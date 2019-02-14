package LibDL.outdated.nn.optimization;

import LibDL.outdated.nn.model.Model;
import LibDL.outdated.nn.model.listeners.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

public class SGDOptimizer extends DefaultOptimizer {

    private IterationListener iterationListener;
    private int iterCount = 0;

    public static void main(String[] args) {

    }

    public void setIterationListener(IterationListener iterationListener) {
        this.iterationListener = iterationListener;
    }

    @Override
    public void optimize(Model model) {
        model.update();
    }

    @Override
    public void optimize(Model model, INDArray input, INDArray target) {
        model.setInput(input);
        model.doForward();
        error = lossFunction.computeGradient(target, model.getPreOutput(), model.getLossActivation(), null).negi();
        model.setError(error);
        model.doBackward();
        model.setLearnRate(learnRate);
        model.update();
        iterCount++;
        if (iterationListener != null) {
            iterationListener.onEvent(model, iterCount);
        }
    }

    @Override
    public int getBatchSize() {
        return 1;
    }

    @Override
    public void setBatchSize(int batchSize) {

    }
}
