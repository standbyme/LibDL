package LibDL.optim;

import LibDL.Tensor.Constant;

public class SGD extends Optimizer {
    private final float lr;

    public SGD(Constant[] params, float lr) {
        super(params);
        this.lr = lr;
    }

    @Override
    public void step(){
        for (Constant param : params) {
            param.value.subi(param.dout.mul(lr));
        }
    }
}
