package com.libDL.nn.layer;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public abstract class DefaultLayer implements Layer {
    protected INDArray epsilon;
    protected IActivation activationFunction;
    protected String name;
    protected boolean hasBias;
    protected INDArray input;
    protected INDArray output;
    protected INDArray error;
    protected double learnRate;
    protected INDArray weight;
    protected INDArray bias;
    protected INDArray weightDiff;
    protected INDArray biasDiff;
    protected INDArray preOutput;

    DefaultLayer() {
    }

    DefaultLayer(String name, IActivation activationFunction, int nInput, int nOutput, boolean hasBias) {
        this(name, activationFunction, nInput, nOutput, hasBias, 0.01);
    }

    DefaultLayer(String name, IActivation activationFunction, int nInput, int nOutput, boolean hasBias, double learnRate) {
        this.name = name;
        this.activationFunction = activationFunction;
        this.hasBias = hasBias;
        this.weight = Nd4j.randn(nOutput, nInput);
        this.learnRate = learnRate;

        if (hasBias) {
            this.bias = Nd4j.randn(nOutput, 1);
        }
    }

    public INDArray getPreOutput() {
        return preOutput;
    }

    public void setPreOutput(INDArray preOutput) {
        this.preOutput = preOutput;
    }

    public INDArray getWeightDiff() {
        return weightDiff;
    }

    public void setWeightDiff(INDArray weightDiff) {
        this.weightDiff = weightDiff;
    }

    public INDArray getBiasDiff() {
        return biasDiff;
    }

    public void setBiasDiff(INDArray biasDiff) {
        this.biasDiff = biasDiff;
    }

    public void setHasBias(boolean hasBias) {
        this.hasBias = hasBias;
    }

    public INDArray getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(INDArray epsilon) {
        this.epsilon = epsilon;
    }

    public INDArray getBias() {
        return bias;
    }

    public void setBias(INDArray bias) {
        this.bias = bias;
    }

    public INDArray getWeight() {
        return weight;
    }

    public void setWeight(INDArray weight) {
        this.weight = weight;
    }

    @Override
    public IActivation getActivationFunction() {
        return activationFunction;
    }

    @Override
    public void setActivationFunction(IActivation activationFunction) {
        this.activationFunction = activationFunction;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public void setName(String name) {
        this.name = name;
    }

    public INDArray getInput() {
        return input;
    }

    public void setInput(INDArray input) {
        this.input = input;
    }

    public INDArray getOutput() {
        return output;
    }

    public void setOutput(INDArray output) {
        this.output = output;
    }

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

}
