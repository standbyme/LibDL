package LibDL.nn;

import LibDL.Tensor.Tensor;
import LibDL.optim.Parameter;
import LibDL.utils.Pair;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;


public abstract class Module {

    public abstract Tensor forward(Tensor input);

    public Tensor apply(Tensor input) {
        return forward(input);
    }

    public final Tensor predict(Tensor input) {        // make them happy
        return apply(input);
    }

    public Parameter[] parameters() {
        ArrayList<Parameter> result = new ArrayList<>();
        ArrayList<Module> moduleList = new ArrayList<>();
        moduleList.add(this);
        while (!moduleList.isEmpty()) {
            ArrayList<Module> subModuleList = new ArrayList<>();
            for(Module m: moduleList) {
                subModuleList.addAll(m.getSubModules().stream()
                        .map((p) -> p.second).collect(Collectors.toList()));

                result.addAll(m.getParameters());
            }
            moduleList = subModuleList;
        }

        return result.toArray(new Parameter[0]);
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder(this.getClass().getSimpleName());
        Collection<Pair<String, Module>> modules = getSubModules();
        if (modules.isEmpty())
            return str.append("()").toString();
        str.append("(\n");
        for (Pair<String, Module> m : getSubModules())
            str.append("  (").append(m.first).append("): ").append(m.second).append("\n");
        str.append(")");
        return str.toString();
    }

    private Collection<Pair<String, Module>> getSubModules() {
        Class<? extends Module> cls = this.getClass();
        Field[] fields = cls.getDeclaredFields();
        List<Pair<String, Module>> modules = new ArrayList<>();
        for (Field f : fields) {
            f.setAccessible(true);
            String name = f.getName();
            Object value = null;
            try {
                value = f.get(this);
            } catch (IllegalAccessException e) {
                // will not happen.
                e.printStackTrace();
            }

            if (value instanceof Module) {
                modules.add(new Pair<>(name, (Module) value));
            } else if (value instanceof Module[]) {
                int counter = 0;
                for (Module module : (Module[]) value) {
                    modules.add(new Pair<>(Integer.toString(counter), module));
                    counter++;
                }
            }
        }
        return modules;
    }

    private Collection<Parameter> getParameters() {
        Class<? extends Module> cls = this.getClass();
        Field[] fields = cls.getDeclaredFields();
        List<Parameter> parameters = new ArrayList<>();
        for (Field f : fields) {
            f.setAccessible(true);
            Object value = null;
            try {
                value = f.get(this);
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            }

            if (value instanceof Parameter) {
                parameters.add((Parameter) value);
            }
        }
        return parameters;
    }

}
