package LibDL;

import LibDL.Tensor.LayerTensor;
import LibDL.Tensor.Tensor;
import LibDL.nn.Linear;
import LibDL.nn.WeightInit;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;

public class Serialization {

    /***
     * module is the Object which you want to save in a file.
     * The Object of module must can be serializable, which means
     * Base class of this Object should implement the interface Serializable.
     * path is a String Object that decides where the file saved in.
     * @param module
     * module is the Object which you want to save in a file.
     * @param path
     * module is the Object which you want to save in a file.
     */
    public static void save(Object module, String path) {
        save(module, path, "binary");
//        try
//        {
//            FileOutputStream fileOut = new FileOutputStream(path);
//            ObjectOutputStream out = new ObjectOutputStream(fileOut);
//            out.writeObject(module);
//            out.close();
//            fileOut.close();
//            System.out.println("Serialized data is saved in " + path);
//        }catch (IOException i)
//        {
//            i.printStackTrace();
//        }
    }

    /***
     *
     * @param module
     * module is the Object which you want to save in a file.
     * @param path
     * module is the Object which you want to save in a file.
     * @param method
     * method is the kind of way to save your module.
     */
    public static void save(Object module, String path, String method) {
        try
        {
            switch (method) {
                case "binary":
                    FileOutputStream fileOut = new FileOutputStream(path);
                    ObjectOutputStream out = new ObjectOutputStream(fileOut);
                    out.writeObject(module);
                    out.close();
                    fileOut.close();
                    System.out.println("Serialized data is saved in " + path);
                    break;
                case "json":
                    ObjectMapper objectMapper = new ObjectMapper();
                    String moduleClassName = module.getClass().getName();
                    if (moduleClassName.endsWith("WeightInit"))
                    {
                        WeightInit temp = (WeightInit) module;
                        objectMapper.writeValue(new File(path), temp);
                        System.out.println("Serialized data is saved in " + path);
                    }
                    else {
                        Tensor temp = (Tensor) module;
                        objectMapper.writeValue(new File(path), temp);
                        System.out.println("Serialized data is saved in " + path);
                    }
                    break;
                default:
                    System.out.println("ERROR:There is no method can match!");
                    break;
            }

        }catch (IOException i)
        {
            i.printStackTrace();
        }
    }

    /***
     *
     * @param path
     * the path of the serializable file.
     * @return Object of module.
     */
    public static Object load(String path) {
        try
        {
            if (path.endsWith("ser")) {
                FileInputStream fileIn = new FileInputStream(path);
                ObjectInputStream in = new ObjectInputStream(fileIn);
                return in.readObject();
            }else if (path.endsWith("json")) {
                ObjectMapper objectMapper = new ObjectMapper();
                return objectMapper.readValue(new File(path), Object.class);
            }else {
                System.out.println("ERROR:The format of serialization is not matched!");
                return null;
            }

        }catch(IOException i)
        {
            i.printStackTrace();
            return null;
        }catch (ClassNotFoundException i)
        {
            return null;
        }
    }
}

