package com.jstarcraft.module.datatran.translator.indexmake;

import com.jstarcraft.module.datatran.entity.FLPair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 在许多情况下 标签对象是无法进行数据运算的类型 需要将其转换成{@link INDArray}类型的数据 该类中的方法提供了其中一种解决方案
 *  将标签对象映射为整数 再将该数转换为行数列数均为1的{@link INDArray}对象
 */
public class IndexMake {
    /**
     * 标签与整数之间存在相互映射的关系 该方法选用了自动生成的映射关系
     * @param data
     * @param <F>
     * @param <L>
     * @return
     */
    public static <F,L> List<FLPair<F,INDArray>> indexMake(List<FLPair<F,L>> data) {
        return indexMake(data,new HashMap<>());
    }

    /**
     * 由于{@link List}&lt;L&gt;类型的对象的{@link List#get(int)}方法 可以从索引数值得到标签对象 所以称之为index2Label
     * <ul><li>如果index2Label参数为空列表 则该方法运行后 该对象中含有缺省生成的index2Label列表</li>
     * <li>如果index2Label参数不为空列表 则它指示了索引数值与标签对象之间的映射关系</li></ul>
     * 由映射关系获得标签对象对应的索引数值 再将索引数值转为{@link INDArray} 返回转换后的{@link List}&lt;{@link FLPair}&lt;F,{@link INDArray}&gt;&gt;对象
     * @param data
     * @param index2Label
     * @param <F> 待转换的{@link FLPair}对象们的特征对象类型
     * @param <L> 待转换的{@link FLPair}对象们的标签对象类型
     * @return
     */
    public static <F,L> List<FLPair<F,INDArray>> indexMake(List<FLPair<F,L>> data,List<L> index2Label) {
        if (index2Label.isEmpty()) index2Label.addAll(getDefaultIndex2Label(data));
        Map<L,Integer> label2Index = calculateLabel2Index(index2Label);
        return indexMake(data,label2Index);
    }

    /**
     * 由于{@link Map}&lt;L,Integer&gt;类型的对象 可以从标签对象获得整数值(索引数值) 所以称之为label2Index
     * <ul><li>如果label2Index参数为空列表 则该方法运行后 该对象中含有缺省生成的label2Index列表</li>
     * <li>如果label2Index参数不为空列表 则它指示了标签对象与索引数值之间的映射关系</li></ul>
     * 由映射关系获得标签对象对应的索引数值 再将索引数值转为{@link INDArray} 返回转换后的{@link List}&lt;{@link FLPair}&lt;F,{@link INDArray}&gt;&gt;对象
     * @param data
     * @param label2Index
     * @param <F> 待转换的{@link FLPair}对象们的特征对象类型
     * @param <L> 待转换的{@link FLPair}对象们的标签对象类型
     * @return
     */
    public static <F,L> List<FLPair<F,INDArray>> indexMake(List<FLPair<F,L>> data,Map<L,Integer> label2Index) {
        if (label2Index.isEmpty()) label2Index.putAll(calculateLabel2Index(getDefaultIndex2Label(data)));
        List<FLPair<F,INDArray>> resData = new ArrayList<>();
        data.forEach(p->resData.add(new FLPair<>(p.getFeature(), Nd4j.ones(1,1).mul(label2Index.get(p.getLabel())))));
        return resData;
    }

    /**
     * 由于{@link List}&lt;L&gt;类型的对象的{@link List#get(int)}方法 可以从索引数值得到标签对象 所以称之为index2Label
     *  该方法为获得缺省的index2Label列表
     * @param data
     * @param <F>
     * @param <L>
     * @return
     */
    public static <F,L> List<L> getDefaultIndex2Label(List<FLPair<F,L>> data) {
        return data.stream().map(FLPair::getLabel).distinct().collect(Collectors.toCollection(ArrayList::new));
    }

    /**
     * 由于{@link Map}&lt;L,Integer&gt;类型的对象 可以从标签对象获得整数值(索引数值) 所以称之为label2Index
     *  该方法为从index2Label列表获得对应的label2Index
     * @param index2Label
     * @param <L>
     * @return
     */
    public static <L> Map<L,Integer> calculateLabel2Index(Collection<L> index2Label) {
        Map<L,Integer> label2Index = new HashMap<>();
        int rank = 0;
        for (L label : index2Label) label2Index.put(label, rank++);
        return label2Index;
    }
}
