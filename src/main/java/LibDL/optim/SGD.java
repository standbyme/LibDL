package LibDL.optim;

import LibDL.Tensor.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class SGD extends Optimizer {
    private final float lr;
    private final float momentum;

    private INDArray[] v;

    public SGD(Variable[] parameters, float lr) {
        this(parameters, lr, 0);
    }

    public SGD(Variable[] parameters, float lr, float momentum) {
        super(parameters);
        this.lr = lr;
        this.momentum = momentum;
        this.v = Arrays.stream(params)
                .map(constant -> Nd4j.zerosLike(constant.data))
                .toArray(INDArray[]::new);
    }

    @Override
    public void step() {
        for (int i = 0; i < params.length; i++) {
            Variable param = params[i];
            v[i].muli(momentum).subi(param.grad.mul(lr));
            param.data.addi(v[i]);
        }
    }
}
