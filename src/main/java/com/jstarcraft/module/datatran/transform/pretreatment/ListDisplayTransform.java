package com.jstarcraft.module.datatran.transform.pretreatment;

import com.jstarcraft.module.datatran.transform.Transform;

import java.util.List;

public class ListDisplayTransform implements Transform<String> {

    private int maxDisplay = 10;
    private boolean typeDisplayed = false;

    public ListDisplayTransform(int maxDisplay) {
        this.maxDisplay = maxDisplay;
    }

    public ListDisplayTransform(int maxDisplay, boolean typeDisplayed) {
        this.maxDisplay = maxDisplay;
        this.typeDisplayed = typeDisplayed;
    }

    @Override
    public String tran(Object in) {
        List list = (List)in;
        StringBuilder builder = new StringBuilder();
        if (list == null) return "null";
        if (list.size() <= maxDisplay) {
            for (Object o : list) {
                if (typeDisplayed)builder.append(o.getClass()).append("\n");
                builder.append(o).append("\n");
            }
        } else {
            int topCount = maxDisplay/2, bottomCount = maxDisplay-topCount-1;
            for (Object o : list.subList(0,topCount)) {
                if (typeDisplayed)builder.append(o.getClass()).append("\n");
                builder.append(o).append("\n");
            }
            builder.append("...\n");
            for (Object o : list.subList(list.size()-bottomCount, list.size())) {
                if (typeDisplayed)builder.append(o.getClass()).append("\n");
                builder.append(o).append("\n");
            }
        }
        return builder.toString();
    }

    public void setMaxDisplay(int maxDisplay) {
        this.maxDisplay = maxDisplay;
    }

    public int getMaxDisplay() {
        return maxDisplay;
    }

    public boolean isTypeDisplayed() {
        return typeDisplayed;
    }

    public void setTypeDisplayed(boolean typeDisplayed) {
        this.typeDisplayed = typeDisplayed;
    }
}
