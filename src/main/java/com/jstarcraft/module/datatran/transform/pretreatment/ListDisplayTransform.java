package com.jstarcraft.module.datatran.transform.pretreatment;

import com.jstarcraft.module.datatran.transform.Transform;

import java.util.List;

/**
 * <p>该变换 输入列表对象 输出字符串对象 为输入的聊表对象的字符串表示</p>
 * <p>在列表长于{@link ListDisplayTransform#maxDisplay}的时候转换结果中只含有列表的开头几个和末尾几个对象的内容 中间用省略号代替</p>
 * <p>在{@link ListDisplayTransform#maxDisplay}为<code>true</code>的时候 转换结果会添加上每个对象的类型</p>
 */
public class ListDisplayTransform implements Transform<List<?>, String> {
    /**
     * 默认为<code>10</code> 在列表长于该值的时候转换结果中只含有列表的开头几个和末尾几个对象的内容 中间用省略号代替
     */
    private int maxDisplay = 10;
    /**
     * 默认为<code>false</code> 在该值为<code>true</code>的时候 转换结果会添加上每个对象的类型
     */
    private boolean typeDisplayed = false;

    public ListDisplayTransform() {}

    public ListDisplayTransform(int maxDisplay) {
        this.maxDisplay = maxDisplay;
    }

    public ListDisplayTransform(int maxDisplay, boolean typeDisplayed) {
        this.maxDisplay = maxDisplay;
        this.typeDisplayed = typeDisplayed;
    }

    /**
     * <p>输入列表对象 返回字符串对象 为输入的聊表对象的字符串表示</p>
     * <p>在列表长于{@link ListDisplayTransform#maxDisplay}的时候转换结果中只含有列表的开头几个和末尾几个对象的内容 中间用省略号代替</p>
     * <p>在<code>typeDisplayed</code>为<code>true</code>的时候 转换结果会添加上每个对象的类型</p>
     * @param in 原始数据
     * @return
     */
    @Override
    public String tran(Object in) {
        List<?> list = (List<?>)in;
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
