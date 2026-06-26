$ PYTHONPATH=. python experiments/e2_accuracy/new_run_e2.py 
Detected domains: ['medical', 'finance', 'math', 'coding']

============================================================
RUNNING: Calibrating adapters using Pico
CMD: /home/administraitor/pd/thesis/.venv/bin/python src/pico.py --adapters_dir adapters --output_dir calibrated-adapters/pico
============================================================
Loading 4 adapters from adapters...
Calibrating 64 layers using Pico...
Saved calibrated adapter to calibrated-adapters/pico/coding
Saved calibrated adapter to calibrated-adapters/pico/finance
Saved calibrated adapter to calibrated-adapters/pico/math
Saved calibrated adapter to calibrated-adapters/pico/medical

============================================================
RUNNING: Calibrating adapters using WBP
CMD: /home/administraitor/pd/thesis/.venv/bin/python src/wbp.py --adapters_dir adapters --output_dir calibrated-adapters/wbp
============================================================
Loading 4 adapters from adapters...
Calibrating 64 layers using WBP (beta=1.0)...
Saved calibrated adapter to calibrated-adapters/wbp/coding
Saved calibrated adapter to calibrated-adapters/wbp/finance
Saved calibrated adapter to calibrated-adapters/wbp/math
Saved calibrated adapter to calibrated-adapters/wbp/medical

============================================================
RUNNING: Merging: no_cal_ta
CMD: /home/administraitor/pd/thesis/.venv/bin/python src/ties.py --adapters_dir adapters --output_file merged-adapters/no_cal_ta.pt --method ta
============================================================
Loading 4 adapters from adapters...
Merging 64 layers using TA...
Saved merged dense updates to merged-adapters/no_cal_ta.pt

============================================================
RUNNING: Merging: no_cal_ties
CMD: /home/administraitor/pd/thesis/.venv/bin/python src/ties.py --adapters_dir adapters --output_file merged-adapters/no_cal_ties.pt --method ties
============================================================
Loading 4 adapters from adapters...
Merging 64 layers using TIES...
Saved merged dense updates to merged-adapters/no_cal_ties.pt

============================================================
RUNNING: Merging: pico_ta
CMD: /home/administraitor/pd/thesis/.venv/bin/python src/ties.py --adapters_dir calibrated-adapters/pico --output_file merged-adapters/pico_ta.pt --method ta
============================================================
Loading 4 adapters from calibrated-adapters/pico...
Merging 64 layers using TA...
Saved merged dense updates to merged-adapters/pico_ta.pt

============================================================
RUNNING: Merging: pico_ties
CMD: /home/administraitor/pd/thesis/.venv/bin/python src/ties.py --adapters_dir calibrated-adapters/pico --output_file merged-adapters/pico_ties.pt --method ties
============================================================
Loading 4 adapters from calibrated-adapters/pico...
Merging 64 layers using TIES...
Saved merged dense updates to merged-adapters/pico_ties.pt

============================================================
RUNNING: Merging: wbp_ta
CMD: /home/administraitor/pd/thesis/.venv/bin/python src/ties.py --adapters_dir calibrated-adapters/wbp --output_file merged-adapters/wbp_ta.pt --method ta
============================================================
Loading 4 adapters from calibrated-adapters/wbp...
Merging 64 layers using TA...
Saved merged dense updates to merged-adapters/wbp_ta.pt

============================================================
RUNNING: Merging: wbp_ties
CMD: /home/administraitor/pd/thesis/.venv/bin/python src/ties.py --adapters_dir calibrated-adapters/wbp --output_file merged-adapters/wbp_ties.pt --method ties
============================================================
Loading 4 adapters from calibrated-adapters/wbp...
Merging 64 layers using TIES...
Saved merged dense updates to merged-adapters/wbp_ties.pt

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
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 291/291 [00:04<00:00, 66.51it/s]
Unsloth: Will load unsloth/Meta-Llama-3.1-8B as a legacy tokenizer.
Loading dense merged updates from merged-adapters/no_cal_ta.pt...
Patched 64 parameters.
[lm_eval.models.huggingface|WARNING]`pretrained` model kwarg is not of type `str`. Many other model arguments may be ignored. Please do not launch via accelerate or use `parallelize=True` if passing an existing model this way.
[lm_eval.models.huggingface|WARNING]Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration
Starting benchmark for tasks: ['gsm8k', 'humaneval', 'finqa', 'medmcqa'] ...
Traceback (most recent call last):
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/lm_eval/tasks/_yaml_loader.py", line 109, in _import_func_in_yml
    module = _load_module_with_cache(rel)
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/lm_eval/tasks/_yaml_loader.py", line 88, in _load_module_with_cache
    spec.loader.exec_module(module)  # type: ignore[arg-type]
  File "<frozen importlib._bootstrap_external>", line 883, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/lm_eval/tasks/humaneval/utils.py", line 10, in <module>
    raise e
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/lm_eval/tasks/humaneval/utils.py", line 5, in <module>
    compute_ = hf_evaluate.load("code_eval")
AttributeError: module 'evaluate' has no attribute 'load'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/administraitor/pd/thesis/experiments/e2_accuracy/new_evaluation.py", line 82, in <module>
    main()
  File "/home/administraitor/pd/thesis/experiments/e2_accuracy/new_evaluation.py", line 63, in main
    results = lm_eval.simple_evaluate(
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/lm_eval/utils.py", line 575, in _wrapper
    return fn(*args, **kwargs)
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/lm_eval/evaluator.py", line 302, in simple_evaluate
    loaded = task_manager.load(tasks)
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/lm_eval/tasks/manager.py", line 208, in load
    obj = self._load_spec(spec) if not isinstance(spec, (Task, Group)) else spec  # type:ignore[invalid-argument-type]
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/lm_eval/tasks/manager.py", line 154, in _load_spec
    return self._factory.build(
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/lm_eval/tasks/_factory.py", line 61, in build
    return self._build_task(entry, overrides)
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/lm_eval/tasks/_factory.py", line 67, in _build_task
    cfg = self._load_full_config(entry, overrides)
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/lm_eval/tasks/_factory.py", line 261, in _load_full_config
    cfg = deepcopy(load_yaml(entry.yaml_path, resolve_func=True))
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/lm_eval/tasks/_yaml_loader.py", line 185, in load_yaml
    cfg = yaml.load(fh, Loader=loader_cls)  # noqa: S506
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/yaml/__init__.py", line 81, in load
    return loader.get_single_data()
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/yaml/constructor.py", line 51, in get_single_data
    return self.construct_document(node)
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/yaml/constructor.py", line 60, in construct_document
    for dummy in generator:
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/yaml/constructor.py", line 413, in construct_yaml_map
    value = self.construct_mapping(node)
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/yaml/constructor.py", line 218, in construct_mapping
    return super().construct_mapping(node, deep=deep)
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/yaml/constructor.py", line 143, in construct_mapping
    value = self.construct_object(value_node, deep=deep)
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/yaml/constructor.py", line 100, in construct_object
    data = constructor(self, node)
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/lm_eval/tasks/_yaml_loader.py", line 22, in ctor
    return _import_func_in_yml(spec, base_dir)
  File "/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/lm_eval/tasks/_yaml_loader.py", line 112, in _import_func_in_yml
    raise AttributeError(
AttributeError: Module '/home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/lm_eval/tasks/humaneval/utils.py' has no function 'pass_at_k' (from YAML in /home/administraitor/pd/thesis/.venv/lib/python3.10/site-packages/lm_eval/tasks/humaneval)

[ERROR] Command failed with exit code 1