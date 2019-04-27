package LibDL.nn;

import LibDL.Tensor.Tensor;
import LibDL.optim.Parameter;

import java.lang.reflect.Field;
import java.util.*;


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
                subModuleList.addAll(m.getSubModules().values());
                result.addAll(m.getParameters());
            }
            moduleList = subModuleList;
        }

        return result.toArray(new Parameter[0]);
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder(this.getClass().getSimpleName());
        Map<String, Module> modules = getSubModules();
        if (modules.isEmpty())
            return str.append("()").toString();
        str.append("(\n");
        for (Map.Entry<String, Module> m : modules.entrySet())
            str.append("  (").append(m.getKey()).append("): ").append(m.getValue()).append("\n");
        str.append(")");
        return str.toString();
    }

    private Map<String, Module> getSubModules() {
        Class<? extends Module> cls = this.getClass();
        Field[] fields = cls.getDeclaredFields();
        Map<String, Module> modules = new LinkedHashMap<>();
        int unnamedCount = 0;
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
                modules.put(name, (Module) value);
            } else if (value instanceof Module[]) {
                for (Module module : (Module[]) value) {
                    modules.put(Integer.toString(unnamedCount), module);
                    unnamedCount++;
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
