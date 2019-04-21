package LibDL.Tensor.Operator;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static java.lang.System.out;

public class AutogradTest {

    @Test
    public void testMulti1() {
        Tensor x = new Variable(Nd4j.create(new double[] {0.5}), true);
        Tensor y = x.mul(5);
        Tensor l = x.add(y);
        out.println(l.out);

        out.println("***Backward***");
        l.dout = Nd4j.create(new double[] {1.0});
        l.backward();
        out.println(x.dout);
        //out.println(y.dout); //1
        //out.println(z.dout); //1
    }

    @Test
    public void testMulti3() {
        Tensor x = new Variable(Nd4j.create(new double[] {0.3}), true);
        Tensor l = x.add(x);
        out.println(l.out);

        out.println("***Backward***");
        l.dout = Nd4j.create(new double[] {1.0});
        l.backward();
        out.println(x.dout);
        //out.println(y.dout); //1
        //out.println(z.dout); //1
    }

    @Test
    public void testMulti2() {
        Tensor x = new Variable(Nd4j.create(new double[] {0.7}), true);
        Tensor y = x.mul(2);
        Tensor z = x.mul(3);
        Tensor l = y.add(z);
        out.println(l.out);

        out.println("***Backward***");
        l.dout = Nd4j.create(new double[] {1.0});
        l.backward();
        out.println(x.dout);
        //out.println(y.dout); //1
        //out.println(z.dout); //1
    }

    @Test
    public void testMulti4() {
        Tensor u = new Variable(Nd4j.create(new double[] {0.7}), true);
        Tensor x = u.mul(1);
        Tensor y = x.mul(2);
        Tensor z = x.mul(3);
        Tensor l = y.add(z);
        out.println(l.out);

        out.println("***Backward***");
        l.dout = Nd4j.create(new double[] {1.0});
        l.backward();
        out.println(x.dout);
        out.println(u.dout); // TODO
    }

}
