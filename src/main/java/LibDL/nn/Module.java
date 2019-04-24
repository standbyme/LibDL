package LibDL.nn;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

public abstract class Module {

    public abstract Tensor forward(Tensor input);

    public Tensor apply(Tensor input) {
        return forward(input);
    }

    public final Tensor predict(Tensor input) {        // make them happy
        return apply(input);
    }

    public Variable[] parameters() {
        Class<? extends Module> cls = this.getClass();
        Field[] fields = cls.getDeclaredFields();
        List<Variable> result = new ArrayList<>();
        List<Module> modules = new ArrayList<>();
        for(Field f: fields) {
            f.setAccessible(true);
            Object value = null;
            try {
                value = f.get(this);
            } catch (IllegalAccessException e) {
                // will not happen.
                e.printStackTrace();
            }

            if(value instanceof Variable) {
                // Add the class's own parameters
                result.add((Variable)value);
            }
            else if(value instanceof Module) {
                modules.add((Module) value);
            }
            else if(value instanceof Module[]) {
                modules.addAll(Arrays.asList((Module[]) value));
            }
        }

        // Add submodules' parameters
        for(Module module: modules)
            result.addAll(module.getDeclaredParameters());

        return result.toArray(new Variable[0]);
    }

    private  Collection<Variable> getDeclaredParameters() {
        Class<? extends Module> cls = this.getClass();
        Field[] fields = cls.getDeclaredFields();
        List<Variable> parameters = new ArrayList<>();
        for(Field f: fields) {
            f.setAccessible(true);
            Object value = null;
            try {
                value = f.get(this);
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            }

            if(value instanceof Variable) {
                parameters.add((Variable)value);
            }
        }
        return parameters;
    }

}
