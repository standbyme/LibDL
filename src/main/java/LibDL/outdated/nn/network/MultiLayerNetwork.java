package LibDL.outdated.nn.network;

import LibDL.outdated.nn.layer.DenseLayer;
import LibDL.outdated.nn.layer.Layer;
import LibDL.outdated.nn.layer.OutputLayer;
import LibDL.outdated.nn.model.listeners.Listener;
import LibDL.outdated.nn.optimization.SGDOptimizer;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;

import java.util.ArrayList;

public class MultiLayerNetwork implements NeuralNetwork {

    private ArrayList<Layer> hiddenLayers;
    private OutputLayer outputLayer;
    private INDArray input;
    private INDArray error;


    public MultiLayerNetwork() {
        hiddenLayers = new ArrayList();
    }

    public static void main(String[] args) {
        double[][] XOR_INPUT = {{0.0, 0.0}, {1.0, 0.0},
                {0.0, 1.0}, {1.0, 1.0}};

        double[][] XOR_IDEAL = {{0.0}, {1.0}, {1.0}, {0.0}};
        MultiLayerNetwork mln = new MultiLayerNetwork();
        DenseLayer l1 = new DenseLayer(2, 4, true);
        DenseLayer lm = new DenseLayer(4, 2, true);
        OutputLayer l2 = new OutputLayer(4, 1, true);
        l1.setActivationFunction(new ActivationReLU());
        l2.setActivationFunction(new ActivationReLU());
        lm.setActivationFunction(new ActivationReLU());
        INDArray w1 = Nd4j.randn(new int[]{3, 2}, 2);
        INDArray w2 = Nd4j.randn(new int[]{1, 3}, 12);
        //l1.setWeight(w1);
        //l2.setWeight(w2);
        mln.addHiddenLayer(l1);
        //mln.addHiddenLayer(lm);
        mln.setOutputLayer(l2);
        SGDOptimizer sgdOptimizer = new SGDOptimizer();
        sgdOptimizer.setLearnRate(0.01);
        sgdOptimizer.setLossFunction(new LossMSE());
        sgdOptimizer.setIterationListener((model, iterCount) -> {
            if (iterCount % 1000 == 0) {
                System.out.println("Error:" + sgdOptimizer.getError());
            }
        });
        int i = 0;
        do {
            sgdOptimizer.optimize(mln, Nd4j.create(XOR_INPUT[i % 4], new int[]{2, 1}), Nd4j.create(XOR_IDEAL[i % 4]));
            i++;
        } while (i < 10000000);

        for (int j = 0; j < 4; j++) {
            INDArray predict = mln.predict(Nd4j.create(XOR_INPUT[j], new int[]{2, 1}));
            System.out.println(predict);
        }
    }

    public void setHiddenLayers(ArrayList<Layer> hiddenLayers) {
        this.hiddenLayers = hiddenLayers;
    }

    public void addHiddenLayer(Layer hiddenLayer) {
        this.hiddenLayers.add(hiddenLayer);
    }

    public void addHiddenLayers(Layer... layers) {
        for (Layer l : layers) {
            hiddenLayers.add(l);
        }
    }

    public void setOutputLayer(OutputLayer outputLayer) {
        this.outputLayer = outputLayer;
    }

    @Override
    public void fit(DataSetIterator iterator) {

    }

//
//    @Override
//    public void setOptimizer(Optimizer optimizer) {
//
//    }
//
//    @Override
//    public Optimizer getOptimizer() {
//        return null;
//    }

    @Override
    public INDArray predict(INDArray input) {
        INDArray lastOut = null;
        for (int i = 0; i < hiddenLayers.size(); i++) {
            Layer layer = hiddenLayers.get(i);
            if (lastOut == null) {
                lastOut = layer.run(input);
            } else {
                lastOut = layer.run(lastOut);
            }
        }
        return outputLayer.run(lastOut);
    }

    @Override
    public void addListener(Listener listener) {

    }

    @Override
    public void doForward() {
        Layer last = null, cur;
        for (int i = 0; i < hiddenLayers.size(); i++) {
            cur = hiddenLayers.get(i);
            if (last != null) {
                cur.setInput(last.getOutput());
            } else {
                cur.setInput(input);
            }
            cur.doForward();
            last = cur;
        }
        outputLayer.setInput(last.getOutput());
        outputLayer.doForward();
    }

    @Override
    public void doBackward() {
        outputLayer.setError(error);
        outputLayer.doBackward();
        Layer last = outputLayer, cur;
        for (int i = hiddenLayers.size() - 1; i >= 0; i--) {
            cur = hiddenLayers.get(i);
            cur.setError(last.getEpsilon());
            cur.doBackward();
            last = cur;
        }
    }

    @Override
    public void update() {
        for (Layer l : hiddenLayers) {
            l.update();
        }
        outputLayer.update();
    }

    @Override
    public void setInput(INDArray input) {
        this.input = input;
    }

    @Override
    public INDArray getOutput() {
        return outputLayer.getOutput();
    }

    @Override
    public IActivation getLossActivation() {
        return outputLayer.getActivationFunction();
    }

    @Override
    public INDArray getPreOutput() {
        return outputLayer.getPreOutput();
    }

    @Override
    public void setError(INDArray error) {
        this.error = error;
    }

    @Override
    public void setLearnRate(double learnRate) {
        for (Layer l : hiddenLayers) {
            l.setLearnRate(learnRate);
        }
        outputLayer.setLearnRate(learnRate);
    }
}
