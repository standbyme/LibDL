package com.jstarcraft.module.image2Vec.flatroot;

import java.io.File;
import java.util.ArrayList;

public class FlatRoot {
    /**
     * 调用了{@link FileFlatRoot#flat(File, String[])}
     * @param rootFile
     * @param allowedExtensions
     * @return
     */
    public static ArrayList<File> fileFR(File rootFile, String[] allowedExtensions) {
        return FileFlatRoot.flat(rootFile, allowedExtensions);
    }

    /**
     * 调用了{@link FlatRoot#fileFR(File, String[])}
     * @param dirName
     * @param allowedExtensions
     * @return
     */
    public static ArrayList<File> fileFR(String dirName, String[] allowedExtensions) {
        return FileFlatRoot.flat(new File(dirName), allowedExtensions);
    }

}
