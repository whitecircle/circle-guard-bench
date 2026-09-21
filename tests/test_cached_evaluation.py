import itertools
import tempfile
import unittest

from datasets import Dataset

from circleguardbench.context import GuardbenchContext
from circleguardbench.evaluator import Evaluator
from circleguardbench.models_config import ModelType
from circleguardbench.storage.leaderboard import load_leaderboard
from circleguardbench.storage.objects import BenchResult, ModelOutput, Verdict
from circleguardbench.storage.results import append_result


GROUPS = {
    "default_prompts": "",
    "jailbreaked_prompts": "jb_",
    "default_answers": "ans_",
    "jailbreaked_answers": "jb_ans_",
}


def example(prompt_hash, safe=False):
    verdict = "safe" if safe else "unsafe"
    return {
        "prompt_hash": prompt_hash,
        "prompt": "example prompt",
        "prompt_verdict": verdict,
        "harm_category": None if safe else "test category",
        "default_answer": "example answer",
        "default_answer_verdict": verdict,
        "jailbreaked_prompt": None if safe else "example jailbreak",
        "jailbreaked_answer": None if safe else "example jailbreak answer",
    }


class CachedEvaluationTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.ctx = GuardbenchContext()
        self.ctx.is_initialized = True
        self.ctx.bench_name = "test"
        self.ctx.results_dir = temp.name
        self.ctx.dataset_categories = ["test category"]
        self.ctx.dataset = Dataset.from_list([example("one"), example("two")])
        self.evaluator = Evaluator(self.ctx, force=False, using_cached=True)

    def cache(self, model_name, hashes, verdict=Verdict.UNSAFE):
        for prompt_hash in hashes:
            append_result(
                self.ctx.bench_name,
                model_name,
                BenchResult(
                    prompt_hash=prompt_hash,
                    result=ModelOutput(verdict=verdict, run_time_ms=1),
                ),
                self.ctx.results_dir,
            )

    def evaluate(self, model_name):
        self.evaluator.evaluate_model(model_name, ModelType.CLASSIFIER.value)
        return load_leaderboard(
            self.ctx.bench_name, self.ctx.results_dir
        ).find_by_model_name(model_name).per_category_metrics

    def test_every_nonempty_combination_of_complete_groups(self):
        for count in range(1, len(GROUPS) + 1):
            for selected in itertools.combinations(GROUPS, count):
                with self.subTest(groups=selected):
                    model = "-".join(selected)
                    for group in selected:
                        self.cache(model, [GROUPS[group] + h for h in ("one", "two")])
                    metrics = self.evaluate(model)["test category"]
                    for group in GROUPS:
                        value = getattr(metrics, group)
                        if group in selected:
                            self.assertEqual(value.sample_count, 2)
                            self.assertEqual(value.accuracy, 1.0)
                        else:
                            self.assertIsNone(value)

    def test_partial_group_still_raises(self):
        for group, prefix in GROUPS.items():
            with self.subTest(group=group):
                self.cache(group, [prefix + "one"])
                with self.assertRaisesRegex(ValueError, "Missing 1 out of 2"):
                    self.evaluate(group)

    def test_safe_metrics_reuse_available_default_group(self):
        self.ctx.dataset = Dataset.from_list([example("safe", safe=True)])
        for group in ("default_prompts", "default_answers"):
            with self.subTest(group=group):
                self.cache(group, [GROUPS[group] + "safe"], Verdict.SAFE)
                metrics = self.evaluate(group)["Safe Prompts"]
                default = getattr(metrics, group)
                sibling = group.replace("default", "jailbreaked")
                self.assertEqual(default.sample_count, 1)
                self.assertEqual(getattr(metrics, sibling), default)


if __name__ == "__main__":
    unittest.main()
