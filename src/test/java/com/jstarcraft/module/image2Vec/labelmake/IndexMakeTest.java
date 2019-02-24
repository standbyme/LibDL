package com.jstarcraft.module.image2Vec.labelmake;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public class IndexMakeTest {

    private List<String> features = Arrays.asList("a", "ab", "abc", "a");

    private Function<String, String> function = new Function<String, String>() {
        @Override
        public String apply(String s) {
            return "label" + s.length();
        }
    };

    private List<String> index2Label_1 = new ArrayList<>();
    /**
     * 0 <-> label2 / 1 <-> label3 / 2 <-> label1
     */
    private List<String> index2Label_2 = Arrays.asList("label2", "label3", "label1");

    @Test
    public void indexMakeTest() {
        System.out.println(IndexMake.indexMake(features, function));

        System.out.println(index2Label_1);
        System.out.println(IndexMake.indexMake(features, function, index2Label_1));
        System.out.println(index2Label_1);

        System.out.println(IndexMake.indexMake(features, function, index2Label_2));
    }
}
