The DPT model uses a **Concolution Decoder** to reconstruct the spatial information from high level features extracted from the transformer encoder.  
In DPT, transformer encoder captures high level, global information but it lacks the capability to maintain the fine-grained spatial details. The convolution decoder address this by reassembling spatial information, progressively refining features to restore spatail resolution for pixel-level predictions. 
