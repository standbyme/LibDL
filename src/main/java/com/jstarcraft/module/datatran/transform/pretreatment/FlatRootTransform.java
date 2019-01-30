package com.jstarcraft.module.datatran.transform.pretreatment;

import com.jstarcraft.module.datatran.transform.Transform;

import java.io.File;
import java.io.FileFilter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;

public class FlatRootTransform implements Transform<ArrayList<File>> {

    private class InQueueFileFilter implements FileFilter {
        String[] allowedExtensions;

        public InQueueFileFilter(String[] allowedExtensions) {
            this.allowedExtensions = allowedExtensions;
        }

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

    private String[] allowedExtensions;

    public FlatRootTransform(String... allowedExtensions) {
        this.allowedExtensions = allowedExtensions;
    }

    @Override
    public ArrayList<File> tran(Object rootFileObject) {
        File rootFile = (File) rootFileObject;
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
