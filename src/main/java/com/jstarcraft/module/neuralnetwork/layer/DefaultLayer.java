package com.jstarcraft.module.neuralnetwork.layer;

import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class DefaultLayer implements BaseLayer{
    private ActivationFunction activationFunction;
    private String name;
    private INDArray input;
    private INDArray output;
    private INDArray error;
    private double learnRate;
    private INDArray weight;
    private INDArray bias;

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
    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    @Override
    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    @Override
    public void setName(String name) {
        this.name = name;
    }

    @Override
    public String getName() {
        return name;
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
