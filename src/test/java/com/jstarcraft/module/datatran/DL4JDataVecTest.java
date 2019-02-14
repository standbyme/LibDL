package com.jstarcraft.module.datatran;


import org.datavec.image.loader.BaseImageLoader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.nd4j.linalg.dataset.api.*;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

public class DL4JDataVecTest {

    //Images are of format given by allowedExtension -
    private static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    // {"tif", "jpg", "png", "jpeg", "bmp", "JPEG", "JPG", "TIF", "PNG"}

    private static final long seed = 12345;

    private static final Random randNumGen = new Random(seed);

    private static final int height = 50;
    private static final int width = 50;
    private static final int channels = 3;

    public static void main(String[] args) throws Exception {
        //nyTest();

        List<Integer> arr = new ArrayList<>();
        for (int i=0;i<10;i++)arr.add(i);
        System.out.println(arr.toString());
        arr = arr.subList(0,5);
        System.out.println(arr.toString());
        arr.add(6);
        System.out.println(arr.toString());
    }

    public static void nyTest() throws Exception {
        System.out.println("start");
        File parentDir = new File("C:/Index/Program/datatran/img");
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 50);
        InputSplit trainData = filesInDirSplit[0];
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        ImageTransform transform = new MultiImageTransform(randNumGen
                //, new FlipImageTransform()
                //, new ShowImageTransform("After transform")
        );
        //ImageTransform transform = new MultiImageTransform(randNumGen,new ShowImageTransform("Display - before "));
        recordReader.initialize(trainData,transform);

        /*
        while(recordReader.hasNext()) {
            System.out.println("-----");
            List<Writable> li = recordReader.next();
            for (int i = 0; i < li.size(); i++) {
                System.out.println(li.get(i).toString());
            }
            System.out.println("-----");
        }
        */

        int outputNum = recordReader.numLabels();
        int batchSize = 10; // Minibatch size. Here: The number of images to fetch for each call to dataIter.next().
        int labelIndex = 1; // Index of the label Writable (usually an IntWritable), as obtained by recordReader.next()
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, outputNum);
        while (dataIter.hasNext()) {
            DataSet ds = dataIter.next();
            System.out.println(ds); // 在命令行中打印结果
            try {
                Thread.sleep(3000);                 //1000 milliseconds is one second.
            } catch(InterruptedException ex) {
                Thread.currentThread().interrupt();
            }
        }

    }
}
