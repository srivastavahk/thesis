$ PYTHONPATH=. python experiments/e2_accuracy/new_run_e2.py 
Detected domains: ['medical', 'finance', 'math', 'coding']

[SKIP] Pico calibration already completed.

[SKIP] WBP calibration already completed.

[SKIP] Merging no_cal_ta already completed.

[SKIP] Merging no_cal_ties already completed.

[SKIP] Merging pico_ta already completed.

[SKIP] Merging pico_ties already completed.

[SKIP] Merging wbp_ta already completed.

[SKIP] Merging wbp_ties already completed.

============================================================
RUNNING: Evaluating: no_cal_ta
CMD: /home/administraitor/pd/thesis/.venv/bin/python experiments/e2_accuracy/new_evaluation.py --base_model unsloth/Meta-Llama-3.1-8B --merged_path merged-adapters/no_cal_ta.pt --output_file results/e2/eval_no_cal_ta.json
============================================================
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
Loading base model unsloth/Meta-Llama-3.1-8B...
==((====))==  Unsloth 2026.6.9: Fast Llama patching. Transformers: 5.5.0.
   \\   /|    Quadro RTX 6000. Num GPUs = 1. Max memory: 23.457 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.10.0+cu130. CUDA: 7.5. CUDA Toolkit: 13.0. Triton: 3.6.0
\        /    Bfloat16 = FALSE. FA [Xformers = 0.0.35. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
[unsloth_zoo.log|WARNING]Device does not support bfloat16. Will change to float16.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 291/291 [00:04<00:00, 58.90it/s]
Unsloth: Will load unsloth/Meta-Llama-3.1-8B as a legacy tokenizer.
Loading dense merged updates from merged-adapters/no_cal_ta.pt...
Patched 64 parameters.
[lm_eval.models.huggingface|WARNING]`pretrained` model kwarg is not of type `str`. Many other model arguments may be ignored. Please do not launch via accelerate or use `parallelize=True` if passing an existing model this way.
[lm_eval.models.huggingface|WARNING]Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration
Starting benchmark for tasks: ['gsm8k', 'humaneval', 'finqa', 'medmcqa'] ...
Traceback (most recent call last):
  File "/home/administraitor/pd/thesis/experiments/e2_accuracy/new_evaluation.py", line 83, in <module>
    main()
  File "/home/administraitor/pd/thesis/experiments/e2_accuracy/new_evaluation.py", line 63, in main
    results = lm_eval.simple_evaluate(
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/lm_eval/utils.py", line 575, in _wrapper
    return fn(*args, **kwargs)
TypeError: simple_evaluate() got an unexpected keyword argument 'include_path'

[ERROR] Command failed with exit code 1
