package com.jstarcraft.module.datatran.translator;

import com.jstarcraft.module.datatran.entity.FLPair;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * 封装了{@link Stream}中的一些函数式编程的方法 并加入一些适用于{@link Translator}类特有的函数式编程方法
 * @param <F>
 * @param <L>
 */
public class TranslatorStream<F,L> {
    /**
     * 被封装的{@link Stream}&lt;{@link FLPair}&lt;F,L&gt;&gt;对象
     */
    private Stream<FLPair<F,L>> stream;

    /**
     * 构造函数
     * @param stream
     */
    public TranslatorStream(Stream<FLPair<F,L>> stream) {
        this.stream = stream;
    }

    /**
     * 传入对每个{@link TranslatorStream#stream}中的{@link FLPair}对象所做的转换 对其进行该转换
     * @param function
     * @param <FO>
     * @param <LO>
     * @return
     */
    public <FO,LO> TranslatorStream<FO,LO> map(Function<FLPair<F,L>,FLPair<FO,LO>> function) {
        return new TranslatorStream<>(stream.map(function));
    }

    /**
     * 传入对每个{@link TranslatorStream#stream}中的{@link FLPair}对象中的特征变量所做的转换 对其进行该转换
     * @param function
     * @param <FO>
     * @return
     */
    public <FO> TranslatorStream<FO,L> mapF(Function<F,FO> function) {
        return new TranslatorStream<>(stream.map(FLPair.forFeature(function)));
    }

    /**
     * 传入对每个{@link TranslatorStream#stream}中的{@link FLPair}对象中的标签变量所做的转换 对其进行该转换
     * @param function
     * @param <LO>
     * @return
     */
    public <LO> TranslatorStream<F,LO> mapL(Function<L,LO> function) {
        return new TranslatorStream<>(stream.map(FLPair.forLabel(function)));
    }

    /**
     * 封装了{@link TranslatorStream#stream}中的{@link Stream#forEach(Consumer)}方法 原理同{@link TranslatorStream#map(Function)}
     * @param consumer
     */
    public void forEach(Consumer<FLPair<F,L>> consumer) {
        stream.forEach(consumer);
    }

    /**
     * 封装了{@link TranslatorStream#stream}中的{@link Stream#forEach(Consumer)}方法 原理同{@link TranslatorStream#mapF(Function)}
     * @param consumer
     */
    public void forEachF(Consumer<F> consumer) {
        stream.forEach(FLPair.forFeature(consumer));
    }

    /**
     * 封装了{@link TranslatorStream#stream}中的{@link Stream#forEach(Consumer)}方法 原理同{@link TranslatorStream#mapL(Function)}
     * @param consumer
     */
    public void forEachL(Consumer<L> consumer) {
        stream.forEach(FLPair.forLabel(consumer));
    }

    /**
     * 封装了{@link TranslatorStream#stream}中的{@link Stream#collect(Collector)}方法
     * @param collector
     * @param <A>
     * @param <R>
     * @return
     */
    public <A,R> Object collect(Collector<FLPair<F,L>,A,R> collector) {
        return stream.collect(collector);
    }

    /**
     * 封装了{@link TranslatorStream#stream}中的{@link Stream#filter(Predicate)}方法
     * @param function
     * @return
     */
    public TranslatorStream<F,L> filter(Predicate<FLPair<F,L>> function) {
        return new TranslatorStream<>(stream.filter(function));
    }

    /**
     * 将该{@link TranslatorStream}对象转换回{@link Translator}对象
     * @return
     */
    public Translator<F,L> collect2Translator() {
        ArrayList<FLPair<F,L>> resData = stream.collect(Collectors.toCollection(ArrayList::new));
        return new Translator<>(resData);
    }

    /**
     * 获取该{@link TranslatorStream}中包含的特征数组
     * @return
     */
    public List<F> getFeatureList() {
        return stream.collect(ArrayList::new,(l,p)->l.add(p.getFeature()),ArrayList::addAll);
    }

    /**
     * 获取该{@link TranslatorStream}中包含的标签数组
     * @return
     */
    public List<L> getLabelList() {
        return stream.collect(ArrayList::new,(l,p)->l.add(p.getLabel()),ArrayList::addAll);
    }

    public Stream<FLPair<F,L>> getStream() {
        return stream;
    }

    public void setStream(Stream<FLPair<F,L>> stream) {
        this.stream = stream;
    }
}
