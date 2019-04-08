package LibDL.outdated.nn.model.listeners;

import LibDL.outdated.nn.model.Model;

public interface IterationListener {
    void onEvent(Model model, int iterCount);
}
