package com.jstarcraft.module.datatran.function.flatroot;

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
     * @param fileName
     * @param allowedExtensions
     * @return
     */
    public static ArrayList<File> fileFR(String fileName, String[] allowedExtensions) {
        return FileFlatRoot.flat(new File(fileName), allowedExtensions);
    }

}
