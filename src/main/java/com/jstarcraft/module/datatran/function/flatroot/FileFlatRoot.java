package com.jstarcraft.module.datatran.function.flatroot;

import java.io.File;
import java.io.FileFilter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;

/**
 * 含有从一个{@link File}对象得到该路径下所有某些类型的文件的方法
 */
public class FileFlatRoot {
    /**
     * 文件筛选器 后缀名为{@link InQueueFileFilter#allowedExtensions}中之一的文件或者文件夹 可以满足筛选条件
     */
    private static class InQueueFileFilter implements FileFilter {
        /**
         * 满足条件的后缀名数组
         */
        String[] allowedExtensions;

        /**
         * 构造函数 设置了满足条件的后缀名数组
         * @param allowedExtensions
         */
        public InQueueFileFilter(String[] allowedExtensions) {
            this.allowedExtensions = allowedExtensions;
        }

        /**
         * 重写了{@link FileFilter}对象判断文件是否满足条件的方法
         * @param pathname
         * @return
         */
        @Override
        public boolean accept(File pathname) {
            if (pathname.isDirectory()) return true;
            boolean match = false;
            for (String allowedExtension : allowedExtensions) {
                if (pathname.getName().endsWith(allowedExtension)) {
                    match = true;
                    break;
                }
            }
            return match;
        }
    }

    /**
     * 使用BFS遍历目录 获得满足{@link FileFilter}对象判断条件的{@link File}对象对象列表
     * @param rootFile 根目录对应的{@link File}对象
     * @return 满足条件的 {@link File}对象列表
     */
    public static ArrayList<File> flat(File rootFile, String[] allowedExtensions) {
        ArrayList<File> res = new ArrayList<>();
        LinkedList<File> queue = new LinkedList<>();
        queue.add(rootFile);
        while (!queue.isEmpty()) {
            File frontFile = queue.removeFirst();
            if (frontFile.isFile()) {
                res.add(frontFile);
            } else if (frontFile.isDirectory()) {
                File[] subFiles = frontFile.listFiles(new InQueueFileFilter(allowedExtensions));
                if (subFiles != null) {
                    queue.addAll(Arrays.asList(subFiles));
                }
            }
        }
        return res;
    }
}
