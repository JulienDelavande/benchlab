**Note on the CSV report**

This CSV reports the results of an experiment evaluating the energy consumption of various text-to-video generation models, selected among the most downloaded on the Hugging Face Hub.

**Experimental setup:**

* All models were run one video at a time on an NVIDIA H100 GPU and 8-core AMD EPYC 7R13 CPU, using Python 3.10 and the latest `diffusers` library ([commit 368958d](https://github.com/huggingface/diffusers/commit/368958df6f79da805bed6178b90bf1ca76e5d57b)).
* For each model, 10 prompts were tested. Each prompt was run with 2 warm-up passes and then 5 generation passes. The reported `duration_generate` and energy values are the mean across these runs.
* `energy_generate_gpu` is measured with CodeCarbon and reflects the GPU usage in kWh. CPU and RAM estimates (`energy_generate_cpu`, `energy_generate_ram`) are approximate, since precise values are not exposed by the hardware vendor.
* For the LTX model, the generation pipeline includes additional `upsample` and `denoise` steps; their durations and energy consumptions are also reported.
* Other columns in the CSV record generation parameters and hardware details. Parameters were mostly left at each model’s recommended defaults; they may vary between models.

For more details on parameters, please refer to each model’s Hugging Face page.

**Reproducibility:**
The code used to run this benchmark is available here: [JulienDelavande/benchlab – videos](https://github.com/JulienDelavande/benchlab/tree/main/videos)
