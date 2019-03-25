package LibDL.utils.data;


import java.util.function.Consumer;

public class DataLoader {
    public DataLoader(Dataset dataset,
                      int batch_size,
                      boolean shuffle,
                      Sampler sampler,
                      Sampler batch_sampler,
                      int num_workers,
                      Consumer collate_fn,
                      boolean pin_memory,
                      boolean drop_last) {

    }
}
