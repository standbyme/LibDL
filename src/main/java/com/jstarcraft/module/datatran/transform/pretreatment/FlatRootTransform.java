package com.jstarcraft.module.datatran.transform.pretreatment;

import com.jstarcraft.module.datatran.transform.BaseTransform;
import com.jstarcraft.module.datatran.transform.Transform;

import java.io.File;
import java.io.FileFilter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;

/**
 * 该变换 传入一个{@link File}对象 得到该目录下所有后缀名为{@link FlatRootTransform#allowedExtensions}中之一的文件所对应的{@link File}对象
 */
public class FlatRootTransform extends BaseTransform<File,ArrayList<File>> {
    /**
     * 文件筛选器 后缀名为{@link InQueueFileFilter#allowedExtensions}中之一的文件或者文件夹 可以满足筛选条件
     */
    private class InQueueFileFilter implements FileFilter {
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
     * 满足条件的后缀名数组
     */
    private String[] allowedExtensions;
    /**
     * 构造函数 设置了满足条件的后缀名数组
     * @param allowedExtensions
     */
    public FlatRootTransform(String... allowedExtensions) {
        this.allowedExtensions = allowedExtensions;
    }

    /**
     * 使用BFS遍历目录 获得满足{@link FileFilter}对象判断条件的{@link File}对象对象列表
     * @param rootFile 根目录对应的{@link File}对象
     * @return 满足条件的 {@link File}对象列表
     */
    @Override
    public ArrayList<File> tran2(File rootFile) {
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

    public String[] getAllowedExtensions() {
        return allowedExtensions;
    }

    public void setAllowedExtensions(String[] allowedExtensions) {
        this.allowedExtensions = allowedExtensions;
    }
}
