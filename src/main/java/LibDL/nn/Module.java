package LibDL.nn;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class Module {

    public Tensor core;

    public abstract Tensor forward(Tensor input);

    public Tensor apply(Tensor input) {
        core = forward(input);
        return core;
    }

    public final Tensor predict(Tensor input) {        // make them happy
        return apply(input);
    }

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
        }

        return list.toArray(new Variable[0]);
    }

}
