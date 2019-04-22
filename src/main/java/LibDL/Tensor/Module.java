package LibDL.Tensor;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class Module extends Tensor {

    private Tensor core;

    public abstract Tensor forward(Tensor input);


    public void setInput(Tensor input) {
        core = forward(input);
        out = core.out;
        requires_grad = core.requires_grad;
    }

    protected Module() {

    }

    @Override
    public void forwardWithInput() {
    }

    @Override
    public void backward() {
        core.dout = dout;
        core.backward();
    }

    public Tensor apply(Tensor input) {
        setInput(input);
        return this;
    }

    public Tensor predict(Tensor input) {        // make them happy
        return apply(input);
    }

    @Override
    public Variable[] parameters() {
        Class<? extends Module> cls = this.getClass();
        Field[] fields = cls.getDeclaredFields();
        List<Variable> list = new ArrayList<>();
        for(Field f: fields) {
            f.setAccessible(true);
            Object value = null;
            try {
                value = f.get(this);
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            }

            if(value instanceof Variable) {
                list.add((Variable)value);
            }
            else if(value instanceof Module) {
                list.addAll(Arrays.asList(((Module) value).parameters()));
            }
            else if(value instanceof Module[]) {
                for(Module module: (Module[])value) {
                    list.addAll(Arrays.asList(module.parameters()));
                }
            }
            // TODO: Support other collections
        }

        return list.toArray(new Variable[0]);
    }

}
