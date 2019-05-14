package LibDL;

import org.jetbrains.annotations.NotNull;

import java.util.LinkedHashMap;

public class OrderedDict<K extends Comparable, V> extends LinkedHashMap<K, V> {
    public OrderedDict() {
        this("Key");
    }

    public OrderedDict(@NotNull String key_description) {
        this.key_description_ = key_description;
    }

    private String key_description_;
}
