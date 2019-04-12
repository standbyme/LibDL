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
//    private final int t;

    private INDArray[] Vdparams, Sdparams, one;
    private INDArray[] beta1_t;
    private INDArray[] beta2_t;

    public Adam(Variable[] params, float lr, float[] betas, double eps) {
        super(params);
        this.lr = lr;
        this.betas = betas;
        this.eps = eps;
        this.Sdparams = Arrays.stream(params)
                .map(constant -> Nd4j.zerosLike(constant.value))
                .toArray(INDArray[]::new);
        this.Vdparams = Arrays.stream(params)
                .map(constant -> Nd4j.zerosLike(constant.value))
                .toArray(INDArray[]::new);
        this.one = Arrays.stream(params)
                .map(constant -> Nd4j.onesLike(constant.value))
                .toArray(INDArray[]::new);
        beta1_t = Arrays.stream(params)
                .map(constant -> Nd4j.onesLike(constant.value))
                .toArray(INDArray[]::new);
        beta2_t = Arrays.stream(params)
                .map(constant -> Nd4j.onesLike(constant.value))
                .toArray(INDArray[]::new);
//        this.t = 0;
    }

    public Adam(Variable[] params, float lr) {
        this(params, lr, new float[]{0.9f, 0.999f}, 1e-8);
    }

    @Override
    public void step() {
        for (int i = 0; i < params.length; i++) {
            Variable param = params[i];
//            System.out.println(Arrays.toString());
            Vdparams[i].muli(betas[0]).addi(param.dout.mul(1 - betas[0]));
            Sdparams[i].muli(betas[1]).addi
                    (param.dout.mul(param.dout).muli(1.0 - betas[1]));
//            correction
            Vdparams[i].divi(beta1_t[i].sub(1).muli(-1));
            Sdparams[i].divi(beta2_t[i].sub(1).muli(-1));
//            update
            beta1_t[i] = beta1_t[i].muli(betas[0]);
            beta2_t[i] = beta2_t[i].muli(betas[1]);
            param.value.subi(Vdparams[i].mul(lr).
                    divi(Transforms.sqrt(Sdparams[i]).add(eps)));
        }

    }
}
