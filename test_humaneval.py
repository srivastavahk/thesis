from datasets import load_dataset
from experiments.e2_accuracy.evaluate import _run_humaneval_problem

ds = load_dataset("openai_humaneval", split="test")
ex = ds[0]

# mock correct completion
mock_completion = ex['prompt'] + "    return [n for n in numbers if sum(map(int, str(abs(n)))) % 3 == 0]"

print("Running test...")
ok = _run_humaneval_problem(mock_completion, ex['test'], ex['entry_point'])
print(f"Result: {ok}")
