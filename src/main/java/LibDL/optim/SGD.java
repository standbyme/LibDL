package LibDL.optim;

import LibDL.Tensor.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class SGD extends Optimizer {
    private final float lr;
    private final float momentum;

    private INDArray[] v;

    public SGD(Parameters parameters, float lr) {
        this(parameters, lr, 0);
    }

    public SGD(Parameters parameters, float lr, float momentum) {
        super(parameters);
        this.lr = lr;
        this.momentum = momentum;
    }

    @Override
    public void step() {
        if(params == null) {
            cacheParams();
            this.v = Arrays.stream(params)
                    .map(constant -> Nd4j.zerosLike(constant.value))
                    .toArray(INDArray[]::new);
        }

        for (int i = 0; i < params.length; i++) {
            Variable param = params[i];
            v[i].muli(momentum).subi(param.grad.mul(lr));
            param.value.addi(v[i]);
        }
    }
}
