package com.jstarcraft.module.datatran.translator.select;

import com.jstarcraft.module.datatran.entity.FLPair;

import java.util.*;

/**
 * 含有一些将{@link List}&lt;{@link FLPair}&lt;F,L&gt;&gt;重组 或者选择其中部分的方法
 */
public class Select {
    /**
     * 从{@link List}&lt;{@link FLPair}&lt;F,L&gt;&gt;中随机抽取一部分
     * @param data
     * @param count 抽取的对象个数
     * @param random
     * @param <F>
     * @param <L>
     * @return
     */
    public static <F,L> List<FLPair<F,L>> sample(List<FLPair<F,L>> data, int count, Random random) {
        Collections.shuffle(data,random);
        data = data.subList(0,count);
        return data;
    }

    /**
     * 将{@link List}&lt;{@link FLPair}&lt;F,L&gt;&gt;打乱
     * @param data
     * @param random
     * @param <F>
     * @param <L>
     * @return
     */
    public static <F,L> List<FLPair<F,L>> shuffle(List<FLPair<F,L>> data,Random random) {
        Collections.shuffle(data,random);
        return data;
    }

    /**
     * 删减{@link List}&lt;{@link FLPair}&lt;F,L&gt;&gt;中的部分{@link FLPair}&lt;F,L&gt; 使得各个标签对象对应的{@link FLPair}&lt;F,L&gt;数目相等
     * @param data
     * @param random
     * @param <F>
     * @param <L>
     * @return
     */
    public static <F,L> List<FLPair<F,L>> balanceLabel(final List<FLPair<F,L>> data,Random random) {
        HashMap<L, ArrayList<Integer>> staMap = new HashMap<>();

        int rank = 0;
        for (FLPair<F,L> pair : data) {
            ArrayList<Integer> staL = staMap.computeIfAbsent(pair.getLabel(), k->new ArrayList<>());
            staL.add(rank ++);
        }

        Integer min = staMap.values().stream()
                .mapToInt(ArrayList::size).min().orElse(0);

        List<FLPair<F,L>> resList = new ArrayList<FLPair<F,L>>();

        staMap.values().forEach(arr->{
            Collections.shuffle(arr, random);
            int m = min;
            for(Integer r : arr) {
                resList.add(data.get(r));
                if (--m == 0)break;
            }
        });
        return resList;
    }
}
