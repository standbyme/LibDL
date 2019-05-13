package LibDL.nn;

import LibDL.OrderedDict;
import LibDL.Tensor;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

public class Module {
    OrderedDict<String, Tensor> buffers_;
    OrderedDict<String, Module> children_;
    boolean is_training_ = true;
    @Nullable String name_;
    OrderedDict<String, Tensor> parameters_;

    public Module() {
        buffers_ = new OrderedDict<>("Parameter");
        buffers_ = new OrderedDict<>("Buffer");
        children_ = new OrderedDict<>("Submodule");
    }

    public Module(@NotNull String name) {
        this.name_ = name;
    }
}
