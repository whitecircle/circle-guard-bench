import json
import logging
import os
import glob
from typing import List, Optional, Dict
from pathlib import Path

import jinja2
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

from circleguardbench.models_config import ModelsRegistry, load_models_configs_registery

logger = logging.getLogger("guardbench.context")


class GuardbenchContext:
    """Context for storing application state between CLI commands."""

    def __init__(self):
        self.models_config: Optional[ModelsRegistry] = None
        self.prompts_templates: Dict[str, jinja2.Template] = {}
        self.dataset: Optional[Dataset] = None
        self.dataset_categories: List[str] = []
        self.is_initialized: bool = False
        self.results_dir: str = ""
        self.bench_name: str = ""

    def _import_datasets(self):
        try:
            import datasets
        except ImportError as exc:
            raise ImportError(
                "Loading a benchmark dataset requires the 'datasets' package: pip install datasets"
            ) from exc
        return datasets

    def load_models(self, models_path: str = "configs/models.json") -> None:
        """
        Load models configuration file.

        Args:
            models_path: Path to the models configuration file

        Raises:
            Exception: If models configuration cannot be loaded
        """
        try:
            self.models_config = load_models_configs_registery(Path(models_path))
            logger.debug(
                f"Successfully loaded {len(self.models_config)} models from configuration"
            )
        except Exception as e:
            logger.error(f"Error loading models configuration: {e}")
            raise Exception(f"Failed to load models configuration: {e}")

    def load_prompt_templates(self, prompts_path: str) -> None:
        """
        Load Jinja2 templates from the prompts directory.

        Args:
            prompts_path: Path to the directory containing prompt templates

        Raises:
            Exception: If required prompt templates are missing
        """
        required_templates = [
            "cot_prompt_eval_regexp",
            "cot_answer_eval_regexp",
            "cot_prompt_eval_so",
            "cot_answer_eval_so",
        ]

        # Create Jinja2 environment
        template_loader = jinja2.FileSystemLoader(prompts_path)
        template_env = jinja2.Environment(loader=template_loader)

        # Find all .jinja files in the prompts directory
        template_files = glob.glob(os.path.join(prompts_path, "*.jinja"))

        if not template_files:
            raise Exception(f"No template files found in {prompts_path}")

        # Load each template
        for template_path in template_files:
            template_name = os.path.basename(template_path).replace(".jinja", "")
            try:
                self.prompts_templates[template_name] = template_env.get_template(
                    f"{template_name}.jinja"
                )
                logger.debug(f"Loaded template: {template_name}")
            except Exception as e:
                logger.warning(f"Failed to load template {template_name}: {e}")

        # Check if required templates are loaded
        missing_templates = [
            t for t in required_templates if t not in self.prompts_templates
        ]
        if missing_templates:
            raise Exception(
                f"Required templates are missing: {', '.join(missing_templates)}"
            )

        logger.info(f"Loaded {len(self.prompts_templates)} prompt templates")

    def load_dataset(self, dataset_path: str) -> None:
        """
        Load the dataset for evaluation.

        Args:
            dataset_path: Path or name of the dataset to load

        Raises:
            Exception: If the dataset cannot be loaded or doesn't have the required format
        """
        try:
            logger.info(f"Loading dataset from: {dataset_path}")
            self.bench_name = dataset_path.split("/")[-1].split(".")[0]

            if os.path.isfile(dataset_path) and dataset_path.endswith(".jsonl"):
                datasets = self._import_datasets()
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = [json.loads(line) for line in f if line.strip()]

                self.dataset = datasets.Dataset.from_list(data)
                logger.info(f"Loaded dataset from local JSONL file: {dataset_path}")
            else:
                datasets = self._import_datasets()
                dataset = datasets.load_dataset(dataset_path)

                if hasattr(dataset, "keys") and "train" in dataset:
                    self.dataset = dataset["train"]
                else:
                    self.dataset = dataset

            required_columns = [
                "harm_category",
                "prompt",
                "prompt_verdict",
                "prompt_hash",
                "default_answer",
                "default_answer_verdict",
                "jailbreaked_prompt",
                "jailbreaked_answer",
            ]
            missing_columns = [
                col for col in required_columns if col not in self.dataset.column_names
            ]

            if missing_columns:
                raise ValueError(
                    f"Dataset is missing required columns: {', '.join(missing_columns)}"
                )

            categories = set(self.dataset["harm_category"])
            self.dataset_categories = [x for x in categories if x is not None]

            num_examples = len(self.dataset)
            logger.info(f"Successfully loaded dataset with {num_examples} examples")

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise Exception(f"Failed to load dataset: {e}")
