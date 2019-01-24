package com.libDL.nn.model.listeners;

import com.libDL.nn.model.Model;

public interface IterationListener {
    void onEvent(Model model, int iterCount);
}
