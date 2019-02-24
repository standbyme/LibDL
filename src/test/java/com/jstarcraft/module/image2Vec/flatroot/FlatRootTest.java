package com.jstarcraft.module.image2Vec.flatroot;

import org.junit.Test;

import java.io.File;
import java.util.List;

public class FlatRootTest {
    @Test
    public void fileFRTest() {
        String[] allowedExtensions = new String[]{"jpg"};
        String dirPath = "F:/Programs/moduleimage/ImageDemo";
        List<File> res = FlatRoot.fileFR(dirPath, allowedExtensions);
        res.forEach(System.out::println);
    }
}
