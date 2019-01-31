package LibDL.nn.optimization;

import LibDL.nn.model.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;

public abstract class DefaultOptimizer implements Optimizer {

    protected ILossFunction lossFunction;

    protected int batchSize;

    protected double learnRate;

    protected INDArray error;

    public INDArray getError() {
        return error;
    }

    public void setError(INDArray error) {
        this.error = error;
    }

    public double getLearnRate() {
        return learnRate;
    }

    public void setLearnRate(double learnRate) {
        this.learnRate = learnRate;
    }

    @Override
    public void optimize(Model model) {

    }

    @Override
    public ILossFunction getLossFunction() {
        return this.lossFunction;
    }

    @Override
    public void setLossFunction(ILossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    @Override
    public int getBatchSize() {
        return batchSize;
    }

    @Override
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }
}
