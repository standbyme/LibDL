package vision.datasets._ImageModule;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.ArrayList;

public class BinaryPattern {
    public static ArrayList<byte[]>[] readPattern(String path, int totalSize, int[] pattern) throws Exception {
        InputStream inStream = null;
        BufferedInputStream bis = null;
        ArrayList<byte[]> bytes = new ArrayList<>();
        ArrayList<byte[]>[] buf = new ArrayList[pattern.length];

        for (int i = 0; i < pattern.length; i++) {
            bytes.add(new byte[pattern[i]]);
        }
        try {
            inStream = new FileInputStream(path);
            bis = new BufferedInputStream(inStream);
            for (int i = 0; i < totalSize; i++) {
                for (int j = 0; j < pattern.length; j++) {
                    bis.read(bytes.get(j));
                    buf[j].add(bytes.get(j).clone());
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (inStream != null) inStream.close();
            if (bis != null) bis.close();
        }
        return buf;
    }
}
