package LibDL.optim;

import LibDL.Tensor.Constant;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class SGD extends Optimizer {
    private final float lr;
    private final float momentum;

    private final INDArray[] v;

    public SGD(Constant[] params, float lr) {
        this(params, lr, 0);
    }

    public SGD(Constant[] params, float lr, float momentum) {
        super(params);
        this.lr = lr;
        this.momentum = momentum;
        this.v = Arrays.stream(params)
                .map(constant -> Nd4j.zerosLike(constant.value))
                .toArray(INDArray[]::new);
    }

    @Override
    public void step() {
        for (int i = 0; i < params.length; i++) {
            Constant param = params[i];
            v[i].muli(momentum).subi(param.dout.mul(lr));
            param.value.addi(v[i]);
        }
    }
}
