package LibDL.optim;

import LibDL.Tensor.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

public class Adam extends Optimizer {

    private final float lr;
    private final float[] betas;
    private final double eps;

    private INDArray[] exp_average_buffers, exp_average_sq_buffers;
    private long[] step_buffers;

    public Adam(Variable[] parameters, float lr, float[] betas, double eps) {
        super(parameters);
        this.lr = lr;
        this.betas = betas;
        this.eps = eps;

        this.exp_average_sq_buffers = Arrays.stream(params)
                .map(constant -> Nd4j.zerosLike(constant.data))
                .toArray(INDArray[]::new);
        this.exp_average_buffers = Arrays.stream(params)
                .map(constant -> Nd4j.zerosLike(constant.data))
                .toArray(INDArray[]::new);
        this.step_buffers = new long[params.length];
    }

    public Adam(Variable[] parameters, float lr) {
        this(parameters, lr, new float[]{0.9f, 0.999f}, 1e-8);
    }

    @Override
    public void step() {

        for (int i = 0; i < params.length; i++) {
            Variable param = params[i];
            exp_average_buffers[i].muli(betas[0]).addi(param.grad.mul(1.0 - betas[0]));
            exp_average_sq_buffers[i].muli(betas[1]).addi
                    (param.grad.mul(param.grad).muli(1.0 - betas[1]));
//            correction
            step_buffers[i] += 1;
            double bias_correction1 = 1 - Math.pow(betas[0], step_buffers[i]);
            double bias_correction2 = 1 - Math.pow(betas[1], step_buffers[i]);
            double step_size = this.lr * Math.sqrt(bias_correction2) / bias_correction1;
            param.data.subi(exp_average_buffers[i].div(Transforms.sqrt(exp_average_sq_buffers[i]).addi(eps)).muli(step_size));
        }

    }
}
