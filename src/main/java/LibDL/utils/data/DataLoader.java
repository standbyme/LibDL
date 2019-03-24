package LibDL.utils.data;


import sun.util.locale.provider.FallbackLocaleProviderAdapter;

import java.util.concurrent.Callable;

public class DataLoader {
    public DataLoader(Dataset dataset,
                      int batch_size,
                      boolean shuffle,
                      Sampler sampler,
                      Sampler batch_sampler,
                      int num_workers,
                      Callable collate_fn,
                      boolean pin_memory,
                      boolean drop_last) {

    }
}
