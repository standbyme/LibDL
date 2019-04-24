package LibDL.nn;

import LibDL.Tensor.Tensor;
import LibDL.optim.Parameter;

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

    public Parameter[] parameters() {
        Class<? extends Module> cls = this.getClass();
        Field[] fields = cls.getDeclaredFields();
        List<Parameter> result = new ArrayList<>();
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

            if(value instanceof Parameter) {
                // Add the class's own parameters
                result.add((Parameter)value);
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

        return result.toArray(new Parameter[0]);
    }

    private  Collection<Parameter> getDeclaredParameters() {
        Class<? extends Module> cls = this.getClass();
        Field[] fields = cls.getDeclaredFields();
        List<Parameter> parameters = new ArrayList<>();
        for(Field f: fields) {
            f.setAccessible(true);
            Object value = null;
            try {
                value = f.get(this);
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            }

            if(value instanceof Parameter) {
                parameters.add((Parameter)value);
            }
        }
        return parameters;
    }

}
