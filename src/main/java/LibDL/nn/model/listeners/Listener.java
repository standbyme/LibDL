package LibDL.nn.model.listeners;

import LibDL.nn.model.Model;

public interface Listener {
    void onEvent(Model model);
}
