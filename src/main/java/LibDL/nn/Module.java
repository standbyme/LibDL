package LibDL.nn;

import LibDL.OrderedDict;
import LibDL.Tensor;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

public class Module {
    @NotNull
    private OrderedDict<String, Tensor> buffers_;
    @NotNull
    private OrderedDict<String, Module> children_;
    private boolean is_training_ = true;
    @Nullable
    private String name_;
    @NotNull
    private OrderedDict<String, Tensor> parameters_;

    public Module() {
        parameters_ = new OrderedDict<>("Parameter");
        buffers_ = new OrderedDict<>("Buffer");
        children_ = new OrderedDict<>("Submodule");
    }

    public Module(@NotNull String name) {
        this.name_ = name;
    }

    private void extend(@NotNull Collection<Tensor> collection, OrderedDict<String, Tensor> dict) {
        collection.addAll(dict.values());
    }


    private String join_name(String name_prefix, String name) {
        int total_size = name.length();
        if (!name_prefix.isEmpty()) {
            total_size += name_prefix.length() + 1;
        }
        StringBuilder full_name = new StringBuilder();
        if (!name_prefix.isEmpty()) {
            full_name.append(name_prefix);
            full_name.append('.');
        }
        full_name.append(name);
        return full_name.toString();
    }


    private void apply_to_submodules(@NotNull BiConsumer<String, Module> function) {
        apply_to_submodules(function, "");
    }

    private void apply_to_submodules(@NotNull BiConsumer<String, Module> function, @NotNull String name_prefix) {
        children_.forEach((key, value) -> {
            var qualified_name = join_name(name_prefix, key);
            function.accept(qualified_name, value);
            value.apply_to_submodules(function, qualified_name);
        });
    }


    Collection<Tensor> parameters() {
        return parameters(true);
    }

    Collection<Tensor> parameters(boolean recurse) {
        if (!recurse) return parameters_.values();
        else {
            ArrayList result = new ArrayList();
            apply(module -> extend(result, module.parameters_));
            return result;
        }
    }

    OrderedDict<String, Tensor> named_parameters(boolean recurse) {
        if (!recurse) return parameters_;
        OrderedDict<String, Tensor> result = new OrderedDict<>();
        apply((String name, Module module) -> module.parameters_.forEach((key, value) -> {
            result.put(join_name(name, key), value);
        }));
        return result;
    }

    private void apply(@NotNull Consumer<Module> function) {
        function.accept(this);
        apply_to_submodules((s, module) -> function.accept(module));
    }

    private void apply(@NotNull BiConsumer<String, Module> function) {
        apply(function, "");
    }

    private void apply(@NotNull BiConsumer<String, Module> function, @NotNull String name_prefix) {
        function.accept(name_prefix, this);
        apply_to_submodules(function, name_prefix);
    }

    Collection<Tensor> buffers(boolean recurse) {
        if (!recurse) return buffers_.values();
        ArrayList<Tensor> result = new ArrayList<>();
        apply(module -> extend(result, module.buffers_));
        return result;
    }

    OrderedDict<String, Tensor> named_buffers(boolean recurse) {
        if (!recurse) return buffers_;
        OrderedDict<String, Tensor> result = new OrderedDict<>();
        apply((String name, Module module) -> module.buffers_.forEach((key, value) -> {
            result.put(join_name(name, key), value);
        }));
        return result;
    }

    Collection<Module> modules(boolean include_self) {
        ArrayList<Module> result = new ArrayList<>();
        if (include_self) {
            apply((Consumer<Module>) result::add);
        } else {
            apply_to_submodules((s, module) -> result.add(module));
        }
        return result;
    }

    OrderedDict<String, Module> named_modules(@NotNull String name_prefix, boolean include_self) {
        OrderedDict<String, Module> result = new OrderedDict<>();
        if (include_self) {
            apply(result::put, name_prefix);
        } else {
            apply_to_submodules(result::put, name_prefix);
        }
        return result;
    }

    Collection<Module> children() {
        return children_.values();
    }

    OrderedDict<String, Module> named_children() {
        return children_;
    }

    private void train(boolean on) {
        children_.forEach((key, value) -> value.train(on));
        is_training_ = on;
    }

    void eval() {
        train(false);
    }

    boolean is_training() {
        return is_training_;
    }

    private void zero_grad() {
        children_.forEach((key, value) -> value.zero_grad());
        parameters_.forEach((key, value) -> {
            var grad = value.grad();
            if (grad.defined()) {
                grad = grad.detach();
                grad.zero_();
            }
        });
    }

    boolean is_serializable() {
        return true;
    }

    Tensor register_parameter(@NotNull String name, @NotNull Tensor tensor) {
        return register_parameter(name, tensor, true);
    }

    private Tensor register_parameter(@NotNull String name, @NotNull Tensor tensor, boolean requires_grad) {
        assert !name.isEmpty();
        assert name.indexOf('.') == -1;

        tensor.set_requires_grad(requires_grad);
        parameters_.put(name, tensor);
        return tensor;
    }

    Tensor register_buffer(@NotNull String name, @NotNull Tensor tensor) {
        assert !name.isEmpty();
        assert name.indexOf('.') == -1;

        buffers_.put(name, tensor);
        return tensor;
    }

    void pretty_print(@NotNull OutputStream stream) throws IOException {
        stream.write(name().getBytes());
    }

    private String name() {
        assert false;
        return null;
    }
}
