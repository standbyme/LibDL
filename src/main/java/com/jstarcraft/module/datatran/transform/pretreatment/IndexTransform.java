package com.jstarcraft.module.datatran.transform.pretreatment;

import com.jstarcraft.module.datatran.transform.BaseTransform;

import java.util.*;

/**
 * <p>该变换 传入一个标签列表 通过{@link IndexTransform#index2Label}指示的索引数值与标签对象的一一对应关系 得到索引列表</p>
 * <p>{@link IndexTransform#index2Label}可以在创建该{@link IndexTransform}对象时指定 也可以缺省添加</p>
 * <p>且如果{@link IndexTransform#index2Label}为空</p>
 * <p>则变换过程中会将缺省生成的对应关系列表加入原{@link IndexTransform#index2Label}列表对象中</p>
 * <p>故如果用户在构造函数{@link IndexTransform#IndexTransform(List)}或者{@link IndexTransform#setIndex2Label(List)}中传入一个空列表对象</p>
 * <p>则该对象在变换执行完成后就被加入了对应关系列表的缺省值</p>
 */
public class IndexTransform extends BaseTransform<ArrayList<?>, ArrayList<Integer>> {
    /**
     * 指示了索引数值与标签对象的一一对应关系 通过{@link List}<code>&lt;?&gt;</code>对象可以从索引数值得到标签对象
     */
    private List<?> index2Label;

    /**
     * 构造函数 {@link IndexTransform#index2Label}被初始化为一个空列表
     */
    public IndexTransform() {
        this.index2Label = new ArrayList<>();
    }

    /**
     * 设置了{@link IndexTransform#index2Label}的构造函数
     * @param index2Label
     */
    public IndexTransform(List<?> index2Label) {
        this.index2Label = index2Label;
    }

    /**
     * <p>变换主过程 过程如下</p>
     * <ul>
     * <li>如果{@link IndexTransform#index2Label}对象为空列表 则在该对象中添加进缺省对应关系列表</li>
     * <li>如果{@link IndexTransform#index2Label}对象不为空列表 则直接以该对象为转换基准</li>
     * </ul>
     * <p>然后以该列表建立的索引与标签的对应关系为转换基准 将传入的标签列表转换为整数列表</p>
     * @param rawLabels
     * @return
     */
    @SuppressWarnings("unchecked")
    @Override
    public ArrayList<Integer> tran2(ArrayList<?> rawLabels) {
        if (index2Label.isEmpty()) {
            index2Label.addAll(getDefaultIndex2Label(rawLabels));
        }

        Map<Object, Integer> label2Index = new HashMap<>();
        int rank = 0;
        for (Object label : index2Label) label2Index.put(label, rank++);

        ArrayList<Integer> indexedLabels = new ArrayList<>();
        rawLabels.forEach(l -> indexedLabels.add(label2Index.get(l)));

        return indexedLabels;
    }

    /**
     * 通过对传入的标签列表去重 得到缺省的对应关系列表
     * @param rawLabels
     * @return
     */
    public List getDefaultIndex2Label(List rawLabels) {
        return Arrays.asList(rawLabels.stream().distinct().toArray());
    }

    public void setIndex2Label(List<?> index2Label) {
        this.index2Label = index2Label;
    }

    public List<?> getIndex2Label() {
        return index2Label;
    }
}
