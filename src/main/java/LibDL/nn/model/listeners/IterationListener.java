package LibDL.nn.model.listeners;

import LibDL.nn.model.Model;

public interface IterationListener {
    void onEvent(Model model, int iterCount);
}
