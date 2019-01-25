package com.libDL.nn.optimization;


import com.libDL.nn.model.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;

public interface Optimizer {
    void optimize(Model model);

    ILossFunction getLossFunction();

    void setLossFunction(ILossFunction lossFunction);


    void optimize(Model model, INDArray input, INDArray output);

    int getBatchSize();

    void setBatchSize(int batchSize);
}
