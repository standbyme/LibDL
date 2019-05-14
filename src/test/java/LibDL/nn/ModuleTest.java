package LibDL.nn;

import LibDL.Tensor;
import LibDL.ld;

class Net extends Module {
    private Tensor W;
    private Tensor b;

    public Net(int N, int M) {
        W = register_parameter("W", ld.randn(new int[]{N, M}));
        b = register_parameter("b", ld.randn(M));
    }

    Tensor forward(Tensor input) {
        return ld.addmm(b, input, W);
    }
}

public class ModuleTest {
    public static void main(String[] args) {
        Net net = new Net(2, 3);
        for (Tensor p : net.parameters()) {
            System.out.println(p);
        }
    }
}