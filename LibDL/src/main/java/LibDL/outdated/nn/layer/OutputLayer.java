package LibDL.outdated.nn.layer;

import LibDL.outdated.nn.model.Model;
import LibDL.outdated.nn.model.listeners.Listener;
import LibDL.outdated.nn.optimization.Optimizer;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;

import java.util.ArrayList;


public class OutputLayer extends DefaultLayer implements Model {

    private ILossFunction lossFunction;
    private int batchSize;
    private ArrayList<Listener> listeners;
    private Optimizer optimizer;

    public OutputLayer(int nIn, int nOut, boolean hasBias) {
        this.hasBias = hasBias;
        weight = Nd4j.randn(new int[]{nOut, nIn}).muli(FastMath.sqrt(2.0 / (nIn + nOut)));
        if (hasBias) {
            bias = Nd4j.zeros(nOut, 1);
        }
    }

    public OutputLayer() {

    }

    @Override
    public void fit(DataSetIterator iterator) {
        while (iterator.hasNext()) {
            DataSet dataSet = iterator.next();
            optimizer.optimize(this, dataSet.getFeatures(), dataSet.getLabels());
        }
    }


    @Override
    public INDArray predict(INDArray input) {
        INDArray weightedInput = weight.mmul(input);
        if (hasBias) {
            weightedInput.addi(bias);
        }
        INDArray output = activationFunction.getActivation(weightedInput, false);
        return output;
    }
//
//    @Override
//    public void setOptimizer(Optimizer optimizer) {
//        this.optimizer = optimizer;
//    }
//
//    @Override
//    public Optimizer getOptimizer() {
//        return optimizer;
//    }

    @Override
    public void addListener(Listener listener) {
        listeners.add(listener);
    }

    @Override
    public IActivation getLossActivation() {
        return activationFunction;
    }

    @Override
    public void doForward() {
        preOutput = weight.mmul(input);
        if (hasBias) {
            preOutput.addi(bias);
        }
        output = activationFunction.getActivation(preOutput, true);
    }

    @Override
    public void doBackward() {
        //INDArray temp = activationFunction.backprop(preOutput, error).getFirst();
        epsilon = weight.transpose().mmul(error);
        weightDiff = error.mmul(input.transpose());
        if (hasBias) {
            biasDiff = getError();
        }
    }

    @Override
    public void update() {
        weight.addi(weightDiff.mul(learnRate));
        if (hasBias) {
            bias.addi(biasDiff.mul(learnRate));
        }
    }

    public INDArray run(INDArray input) {
        INDArray weightedInput = getWeight().mmul(input).add(getBias()); //W (3, 2)
        INDArray output = this.getActivationFunction().getActivation(weightedInput, false);
        return output;
    }

    @Override
    public void setInput(INDArray input) {
        super.setInput(input);
    }
}
