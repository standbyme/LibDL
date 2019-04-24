package LibDL.optim;

import LibDL.Tensor.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

public class RMSProp extends Optimizer {

    private INDArray[] Sdparams;
    private float beta, alpha;
    private double eps;

    public RMSProp(Variable[] parameters, float lr, float alpha, double eps) {
        super(parameters);
        this.alpha = lr;
        this.beta = alpha;
        this.eps = eps;
        this.Sdparams = Arrays.stream(params)
                .map(constant -> Nd4j.zerosLike(constant.data))
                .toArray(INDArray[]::new);
    }


    @Override
    public void step() {
        for (int i = 0; i < params.length; i++) {
            Variable param = params[i];
            Sdparams[i].muli(beta).addi
                    (param.grad.mul(param.grad).muli(1.0 - beta));
            param.data.subi(param.grad.mul(alpha).divi(Transforms.sqrt(Sdparams[i]).add(eps)));
        }
    }
}
