from abc import ABC, abstractmethod
import asyncio
import logging
from typing import List, Dict, Callable, TypeVar
HAVE_TENACITY = True
try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    HAVE_TENACITY = False

import jinja2
from pydantic import BaseModel, Field
import tqdm

import aiohttp
import json

from circleguardbench.models_config import ModelConfig, ModelType
from circleguardbench.providers.utils import (
    DialogMessage,
    generate_openai_response,
    parse_verdict,
    select_template,
)
from circleguardbench.storage.objects import ModelOutput, Verdict


from functools import wraps
import time

T = TypeVar("T")

logger = logging.getLogger(__name__)


class BaseModelProvider(ABC):
    def __init__(
        self, model_config: ModelConfig, prompts_templates: Dict[str, jinja2.Template]
    ):
        self.model_config = model_config
        self.prompt_check_template = select_template(
            prompts_templates, model_config, "prompt"
        )
        self.answer_check_template = select_template(
            prompts_templates, model_config, "answer"
        )
        self.prompts_templates = prompts_templates

    def _measure_time(self, func: Callable[..., T]) -> T:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> ModelOutput:
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                run_time_ms = int((time.perf_counter() - start_time) * 1000)
                if isinstance(result, ModelOutput):
                    result.run_time_ms = run_time_ms
                return result
            except Exception as e:
                run_time_ms = int((time.perf_counter() - start_time) * 1000)
                logger.warning(
                    f"Error in evaluation {self.model_config.name}: {func.__name__}: {str(e)}",
                    exc_info=True,
                )
                return ModelOutput(
                    verdict=Verdict.ERROR, error_message=str(e), run_time_ms=run_time_ms
                )

        return wrapper

    def _retry_decorator(self, func: Callable[..., T]) -> T:
        if not HAVE_TENACITY:
            return func

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((Exception,)),
            reraise=True,
            before_sleep=lambda retry_state: logger.warning(
                f"Retry attempt {retry_state.attempt_number} for {func.__name__}, "
                f"error: {retry_state.outcome.exception()}"
            ),
        )
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        return wrapper

    async def get_prompt_verdict(self, dialog: List[DialogMessage]) -> ModelOutput:
        return await self._measure_time(
            self._retry_decorator(self._get_prompt_verdict)
        )(dialog)

    async def get_answer_verdict(self, dialog: List[DialogMessage]) -> ModelOutput:
        return await self._measure_time(
            self._retry_decorator(self._get_answer_verdict)
        )(dialog)

    @abstractmethod
    async def _get_prompt_verdict(self, dialog: List[DialogMessage]) -> ModelOutput:
        pass

    @abstractmethod
    async def _get_answer_verdict(self, dialog: List[DialogMessage]) -> ModelOutput:
        pass

    async def parallel_prompt_verdicts(
        self,
        dialogs: List[List[DialogMessage]],
        max_concurrent: int = 50,
    ) -> List[ModelOutput]:
        results = [None] * len(dialogs)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single(i: int):
            async with semaphore:
                results[i] = await self.get_prompt_verdict(dialogs[i])

        tasks = [process_single(i) for i in range(len(dialogs))]
        for task in tqdm.tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Processing prompt verdicts",
        ):
            await task
        return results

    async def parallel_answer_verdicts(
        self,
        dialogs: List[List[DialogMessage]],
        max_concurrent: int = 50,
    ) -> List[ModelOutput]:
        results = [None] * len(dialogs)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single(i: int):
            async with semaphore:
                results[i] = await self.get_answer_verdict(dialogs[i])

        tasks = [process_single(i) for i in range(len(dialogs))]
        for task in tqdm.tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Processing answer verdicts",
        ):
            await task
        return results


class OpenAIRegexpModelProvider(BaseModelProvider):
    def __init__(
        self, model_config: ModelConfig, prompts_templates: Dict[str, jinja2.Template]
    ):
        super().__init__(model_config, prompts_templates)
        self.api_model_name = model_config.params.api_model_name
        self.endpoint = model_config.params.endpoint
        self.api_key = model_config.params.api_key

    async def _get_prompt_verdict(self, dialog: List[DialogMessage]) -> ModelOutput:
        rendered_prompt = self.prompt_check_template.render(dialog=dialog)

        answer = await generate_openai_response(
            model=self.api_model_name,
            user_message=rendered_prompt,
            max_tokens=2048,
            temperature=0.1,  # hardcoded
            endpoint=self.endpoint,
            api_key=self.api_key,
        )

        verdict, error_message = parse_verdict(answer, self.model_config.type, "prompt")

        if verdict == Verdict.ERROR:
            logger.warning(
                f"Error parsing answer for model {self.model_config.name}: {error_message}"
            )
            return ModelOutput(verdict=verdict, raw_prediction=answer, error_message=error_message)

        return ModelOutput(verdict=verdict, raw_prediction=answer)

    async def _get_answer_verdict(self, dialog: List[DialogMessage]) -> ModelOutput:
        rendered_prompt, rendered_answer = (
            self.answer_check_template.render(dialog=dialog),
            None,
        )
        if self.model_config.type == ModelType.LLAMA_GUARD:
            rendered_prompt = self.prompt_check_template.render(dialog=dialog)
            rendered_answer = self.answer_check_template.render(dialog=dialog)

        answer = await generate_openai_response(
            model=self.api_model_name,
            user_message=rendered_prompt,
            assistant_message=rendered_answer,
            max_tokens=2048,
            temperature=0.1,  # hardcoded
            endpoint=self.endpoint,
            api_key=self.api_key,
        )

        verdict, error_message = parse_verdict(answer, self.model_config.type, "answer")

        if verdict == Verdict.ERROR:
            logger.warning(
                f"Error parsing answer for model {self.model_config.name}: {error_message}"
            )
            return ModelOutput(verdict=verdict, raw_prediction=answer, error_message=error_message)

        return ModelOutput(verdict=verdict, raw_prediction=answer)


class OpenAIStructuredOutputsModelProvider(BaseModelProvider):
    class SafetyExamination(BaseModel):
        reasoning: str = Field(description="Deep analysis of safety and user's intent")
        verdict: Verdict = Field(
            description="Final verdict based on analysis, 'safe' or 'unsafe'"
        )

    def __init__(
        self, model_config: ModelConfig, prompts_templates: Dict[str, jinja2.Template]
    ):
        super().__init__(model_config, prompts_templates)
        self.api_model_name = model_config.params.api_model_name
        self.endpoint = model_config.params.endpoint
        self.api_key = model_config.params.api_key

    async def _get_prompt_verdict(self, dialog: List[DialogMessage]) -> ModelOutput:
        rendered_prompt = self.prompt_check_template.render(dialog=dialog)

        answer = await generate_openai_response(
            model=self.api_model_name,
            user_message=rendered_prompt,
            max_tokens=2048,
            response_format=self.SafetyExamination,
            temperature=0.1,  # hardcoded
            endpoint=self.endpoint,
            api_key=self.api_key,
        )

        if isinstance(answer, self.SafetyExamination):
            raw_prediction, verdict = answer.reasoning, answer.verdict
        else:
            logger.warning(
                f"Failed to extract SafetyExamination object for model {self.model_config.name}. Answer type: {type(answer)}"
            )
            return ModelOutput(
                verdict=Verdict.ERROR,
                error_message="Cant extract SafetyExamination object",
            )

        return ModelOutput(verdict=verdict, raw_prediction=raw_prediction)

    async def _get_answer_verdict(self, dialog: List[DialogMessage]) -> ModelOutput:
        rendered_prompt = self.answer_check_template.render(dialog=dialog)

        answer = await generate_openai_response(
            model=self.api_model_name,
            user_message=rendered_prompt,
            max_tokens=2048,
            response_format=self.SafetyExamination,
            temperature=0.1,  # hardcoded
            endpoint=self.endpoint,
            api_key=self.api_key,
        )

        if isinstance(answer, self.SafetyExamination):
            raw_prediction, verdict = answer.reasoning, answer.verdict
        else:
            return ModelOutput(
                verdict=Verdict.ERROR,
                error_message="Cant extract SafetyExamination object",
            )

        return ModelOutput(verdict=verdict, raw_prediction=raw_prediction)


class OpenAIModerationModelProvider(BaseModelProvider):
    def __init__(
        self, model_config: ModelConfig, prompts_templates: Dict[str, jinja2.Template]
    ):
        super().__init__(model_config, prompts_templates)
        self.api_model_name = (
            model_config.params.api_model_name or "text-moderation-latest"
        )
        self.endpoint = (
            model_config.params.endpoint or "https://api.openai.com/v1/moderations"
        )
        self.api_key = model_config.params.api_key

    async def _moderate_text(self, text: str) -> ModelOutput:
        """Core moderation function that calls OpenAI API"""
        if not text:
            logger.warning(
                f"Empty text provided for moderation in model {self.model_config.name}"
            )
            return ModelOutput(
                verdict=Verdict.ERROR, error_message="No text provided for moderation"
            )

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            payload = {"input": text, "model": self.api_model_name}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint, headers=headers, json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return ModelOutput(
                            verdict=Verdict.ERROR,
                            error_message=f"API error: {response.status}, {error_text}",
                        )

                    result = await response.json()

            # Process moderation results
            if not result.get("results"):
                logger.warning(
                    f"Moderation did not return results for text: {text[:30]}..."
                )
                return ModelOutput(
                    verdict=Verdict.ERROR,
                    error_message="No moderation results returned",
                )

            moderation_result = result["results"][0]
            flagged = moderation_result.get("flagged", False)

            # Format detailed moderation report
            raw_prediction = json.dumps(
                {
                    "flagged": flagged,
                    "categories": moderation_result.get("categories", {}),
                    "category_scores": moderation_result.get("category_scores", {}),
                },
                indent=2,
            )

            verdict = Verdict.UNSAFE if flagged else Verdict.SAFE

            return ModelOutput(verdict=verdict, raw_prediction=raw_prediction)

        except Exception as e:
            logger.warning(f"Moderation API error: {str(e)}", exc_info=True)
            return ModelOutput(
                verdict=Verdict.ERROR, error_message=f"Moderation API error: {str(e)}"
            )

    async def _get_prompt_verdict(self, dialog: List[DialogMessage]) -> ModelOutput:
        try:
            rendered_text = self.prompt_check_template.render(dialog=dialog)

            if not rendered_text.strip():
                logger.warning(
                    "Empty rendered text detected in answer verdict evaluation"
                )
                return ModelOutput(
                    verdict=Verdict.ERROR,
                    error_message="No text to moderate after template rendering",
                )

            return await self._moderate_text(rendered_text)

        except Exception as e:
            return ModelOutput(
                verdict=Verdict.ERROR,
                error_message=f"Template rendering error: {str(e)}",
            )

    async def _get_answer_verdict(self, dialog: List[DialogMessage]) -> ModelOutput:
        try:
            rendered_text = self.answer_check_template.render(dialog=dialog)

            if not rendered_text.strip():
                logger.warning(
                    "Empty rendered text detected in answer verdict evaluation"
                )
                return ModelOutput(
                    verdict=Verdict.ERROR,
                    error_message="No text to moderate after template rendering",
                )

            return await self._moderate_text(rendered_text)

        except Exception as e:
            return ModelOutput(
                verdict=Verdict.ERROR,
                error_message=f"Template rendering error: {str(e)}",
            )


class VLLMModelProvider(BaseModelProvider):
    def __init__(
        self, model_config: ModelConfig, prompts_templates: Dict[str, jinja2.Template]
    ):
        super().__init__(model_config, prompts_templates)

        try:
            from vllm import AsyncLLMEngine, SamplingParams
            from vllm.engine.arg_utils import AsyncEngineArgs

            self.SamplingParams = SamplingParams
        except ImportError:
            raise ImportError(
                "Failed to import vLLM library. "
                "Please install it using the command: pip install vllm"
            )

        engine_args = AsyncEngineArgs(**model_config.params)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        if hasattr(self.engine, "log_requests"):
            self.engine.log_requests = False

    async def _get_prompt_verdict(self, dialog: List[DialogMessage]) -> ModelOutput:
        try:
            rendered_prompt = self.prompt_check_template.render(dialog=dialog)

            tokenizer = await self.engine.get_tokenizer()
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": rendered_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

            sampling_params = self.SamplingParams(
                temperature=self.model_config.params.get("temperature", 0.1),
                top_p=self.model_config.params.get("top_p", 0.95),
                max_tokens=self.model_config.params.get("max_tokens", 3072),
            )

            final_output = None
            async for output in self.engine.generate(
                formatted_prompt, sampling_params, f"prompt_eval_{time.time()}"
            ):
                final_output = output

            if not final_output:
                logger.warning(
                    f"No response received from model {self.model_config.name} during prompt evaluation"
                )
                return ModelOutput(
                    verdict=Verdict.ERROR, error_message="No response from model"
                )

            answer = final_output.outputs[0].text
            verdict, error_message = parse_verdict(
                answer, self.model_config.type, "prompt"
            )

            if verdict == Verdict.ERROR:
                logger.warning(
                    f"Error parsing answer for model {self.model_config.name}: {error_message}"
                )
                return ModelOutput(verdict=verdict, raw_prediction=answer, error_message=error_message)

            return ModelOutput(verdict=verdict, raw_prediction=answer)

        except Exception as e:
            return ModelOutput(
                verdict=Verdict.ERROR, error_message=f"vLLM error: {str(e)}"
            )

    async def _get_answer_verdict(self, dialog: List[DialogMessage]) -> ModelOutput:
        try:
            rendered_prompt, rendered_answer = (
                self.answer_check_template.render(dialog=dialog),
                None,
            )
            if self.model_config.type == ModelType.LLAMA_GUARD:
                rendered_prompt = self.prompt_check_template.render(dialog=dialog)
                rendered_answer = self.answer_check_template.render(dialog=dialog)

            tokenizer = await self.engine.get_tokenizer()

            formatted_dialog = [{"role": "user", "content": rendered_prompt}]
            if self.model_config.type == ModelType.LLAMA_GUARD:
                formatted_dialog += [{"role": "assistant", "content": rendered_answer}]

            formatted_prompt = tokenizer.apply_chat_template(
                formatted_dialog, tokenize=False, add_generation_prompt=True
            )

            sampling_params = self.SamplingParams(
                temperature=self.model_config.params.get("temperature", 0.1),
                top_p=self.model_config.params.get("top_p", 0.95),
                max_tokens=self.model_config.params.get("max_tokens", 3072),
            )

            final_output = None
            async for output in self.engine.generate(
                formatted_prompt, sampling_params, f"answer_eval_{time.time()}"
            ):
                final_output = output

            if not final_output:
                logger.warning(
                    f"No response received from model {self.model_config.name} during prompt evaluation"
                )
                return ModelOutput(
                    verdict=Verdict.ERROR, error_message="No response from model"
                )

            answer = final_output.outputs[0].text
            verdict, error_message = parse_verdict(
                answer, self.model_config.type, "answer"
            )

            if verdict == Verdict.ERROR:
                logger.warning(
                    f"Error parsing answer for model {self.model_config.name}: {error_message}"
                )
                return ModelOutput(verdict=verdict, raw_prediction=answer, error_message=error_message)

            return ModelOutput(verdict=verdict, raw_prediction=answer)

        except Exception as e:
            return ModelOutput(
                verdict=Verdict.ERROR, error_message=f"vLLM error: {str(e)}"
            )

    def __del__(self):
        if hasattr(self, "engine"):
            self.engine.shutdown()


class SGLangModelProvider(BaseModelProvider):
    def __init__(
        self, model_config: ModelConfig, prompts_templates: Dict[str, jinja2.Template]
    ):
        super().__init__(model_config, prompts_templates)

        try:
            import sglang as sgl

            self.sgl = sgl
        except ImportError:
            raise ImportError(
                "Failed to import sglang library. "
                "Please install it using the command: pip install sglang"
            )

        self.engine = self.sgl.Engine(**model_config.params)

        self.default_sampling_params = {
            "temperature": model_config.params.get("temperature", 0.1),
            "top_p": model_config.params.get("top_p", 0.95),
            "max_new_tokens": model_config.params.get("max_new_tokens", 2048),
        }

        if self.model_config.type == ModelType.LLAMA_GUARD:
            logger.warning(
                "Sglang does not support splitting messages into roles, so assistant message wll be without user message for answers evaluation!"
            )

    async def _get_prompt_verdict(self, dialog: List[DialogMessage]) -> ModelOutput:
        rendered_prompt = self.prompt_check_template.render(dialog=dialog)

        try:
            outputs = await self.engine.async_generate(
                [rendered_prompt], self.default_sampling_params
            )
            answer = outputs[0]["text"]

            verdict, error_message = parse_verdict(
                answer, self.model_config.type, "prompt"
            )

            if verdict == Verdict.ERROR:
                logger.warning(
                    f"Error parsing answer for model {self.model_config.name}: {error_message}"
                )
                return ModelOutput(verdict=verdict, raw_prediction=answer, error_message=error_message)

            return ModelOutput(verdict=verdict, raw_prediction=answer)

        except Exception as e:
            return ModelOutput(
                verdict=Verdict.ERROR,
                error_message=f"SGLang generation error: {str(e)}",
            )

    async def _get_answer_verdict(self, dialog: List[DialogMessage]) -> ModelOutput:
        rendered_prompt = self.answer_check_template.render(dialog=dialog)

        try:
            outputs = await self.engine.async_generate(
                [rendered_prompt], self.default_sampling_params
            )
            answer = outputs[0]["text"]

            verdict, error_message = parse_verdict(
                answer, self.model_config.type, "answer"
            )

            if verdict == Verdict.ERROR:
                logger.warning(
                    f"Error parsing answer for model {self.model_config.name}: {error_message}"
                )
                return ModelOutput(verdict=verdict, raw_prediction=answer, error_message=error_message)

            return ModelOutput(verdict=verdict, raw_prediction=answer)

        except Exception as e:
            return ModelOutput(
                verdict=Verdict.ERROR,
                error_message=f"SGLang generation error: {str(e)}",
            )

    def __del__(self):
        if hasattr(self, "engine"):
            self.engine.shutdown()


class ClassifierModelProvider(BaseModelProvider):
    """
    A generic provider for classification models that can evaluate prompts and answers.
    This provider lazily imports torch and transformers only when needed.
    """

    def __init__(
        self, model_config: ModelConfig, prompts_templates: Dict[str, jinja2.Template]
    ):
        super().__init__(model_config, prompts_templates)

        # Store configuration parameters
        self.model_id = model_config.params.get("model_id")
        self.device = model_config.params.get("device", "cpu")
        self.temperature = model_config.params.get("temperature", 1.0)
        self.max_length = model_config.params.get("max_length", 512)

        # Class mapping configuration
        self.class_to_verdict = model_config.params.get("class_to_verdict", {})
        self.default_verdict = model_config.params.get("default_verdict", "error")

        # Different verdict logic for prompts vs answers
        self.prompt_verdict_logic = model_config.params.get("prompt_verdict_logic", {})
        self.answer_verdict_logic = model_config.params.get("answer_verdict_logic", {})

        # Lazy initialization of model and tokenizer
        self._model = None
        self._tokenizer = None
        self._id2label = None

    def _load_model_and_tokenizer(self):
        """
        Lazily load the model and tokenizer only when needed.
        This ensures torch and transformers are only imported if this provider is used.
        """
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id
            )
            self._model.to(self.device)

            self._id2label = self._model.config.id2label

            # If no class_to_verdict mapping provided, create default mapping
            if not self.class_to_verdict:
                # Default mapping: assume "safe" classes contain words like "safe", "benign", etc.
                for label in self._id2label.values():
                    label_lower = label.lower()
                    if any(
                        safe_word in label_lower
                        for safe_word in ["safe", "benign", "ok", "good"]
                    ):
                        self.class_to_verdict[label] = "safe"
                    else:
                        self.class_to_verdict[label] = "unsafe"

        except ImportError as e:
            raise ImportError(
                f"Failed to import required libraries: {str(e)}. "
                "Please install them using: pip install torch transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Error loading model and tokenizer: {str(e)}")

    async def _get_class_probabilities(self, text: str) -> Dict[str, float]:
        self._load_model_and_tokenizer()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_model, text)

    def _run_model(self, text: str) -> Dict[str, float]:
        import torch
        from torch.nn.functional import softmax

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.inference_mode():
            logits = self._model(**inputs).logits

        scaled_logits = logits / self.temperature

        probabilities = softmax(scaled_logits, dim=-1)
        probs_numpy = probabilities[0].cpu().numpy()

        return {self._id2label[i]: float(prob) for i, prob in enumerate(probs_numpy)}

    async def _classify_text(self, text: str, message_type: str) -> ModelOutput:
        try:
            # Get class probabilities
            class_probs = await self._get_class_probabilities(text)

            # Determine class with highest probability
            predicted_class = max(class_probs.items(), key=lambda x: x[1])[0]

            raw_prediction = {"class": predicted_class, "probabilities": class_probs}

            # Determine verdict based on message type
            if message_type == "prompt":
                verdict_logic = self.prompt_verdict_logic or self.class_to_verdict
            else:  # answer
                verdict_logic = self.answer_verdict_logic or self.class_to_verdict

            # Get verdict from mapping or use default
            verdict_str = verdict_logic.get(predicted_class, self.default_verdict)

            # Convert string verdict to Verdict enum
            if verdict_str.lower() == "safe":
                verdict = Verdict.SAFE
            elif verdict_str.lower() == "unsafe":
                verdict = Verdict.UNSAFE
            else:
                verdict = Verdict.ERROR

            return ModelOutput(verdict=verdict, raw_prediction=str(raw_prediction))

        except Exception as e:
            logger.exception(f"Classification error: {str(e)}")
            return ModelOutput(
                verdict=Verdict.ERROR, error_message=f"Classification error: {str(e)}"
            )

    async def _get_prompt_verdict(self, dialog: List[DialogMessage]) -> ModelOutput:
        try:
            rendered_text = self.prompt_check_template.render(dialog=dialog)

            if not rendered_text.strip():
                logger.warning(
                    f"Empty text after template rendering for model {self.model_config.name}"
                )
                return ModelOutput(
                    verdict=Verdict.ERROR,
                    error_message="No text to classify after template rendering",
                )

            return await self._classify_text(rendered_text, "prompt")

        except Exception as e:
            return ModelOutput(
                verdict=Verdict.ERROR,
                error_message=f"Template rendering error: {str(e)}",
            )

    async def _get_answer_verdict(self, dialog: List[DialogMessage]) -> ModelOutput:
        try:
            rendered_text = self.answer_check_template.render(dialog=dialog)

            if not rendered_text.strip():
                logger.warning(
                    f"Empty text after template rendering for model {self.model_config.name}"
                )
                return ModelOutput(
                    verdict=Verdict.ERROR,
                    error_message="No text to classify after template rendering",
                )

            return await self._classify_text(rendered_text, "answer")

        except Exception as e:
            return ModelOutput(
                verdict=Verdict.ERROR,
                error_message=f"Template rendering error: {str(e)}",
            )


def register_providers():
    from circleguardbench.providers import MODEL_PROVIDERS_REGISTRY

    MODEL_PROVIDERS_REGISTRY.register("openai_regexp")(OpenAIRegexpModelProvider)
    MODEL_PROVIDERS_REGISTRY.register("openai_structured")(
        OpenAIStructuredOutputsModelProvider
    )
    MODEL_PROVIDERS_REGISTRY.register("vllm_regexp")(VLLMModelProvider)
    MODEL_PROVIDERS_REGISTRY.register("sglang_regexp")(SGLangModelProvider)
    MODEL_PROVIDERS_REGISTRY.register("openai_moderation")(
        OpenAIModerationModelProvider
    )
    MODEL_PROVIDERS_REGISTRY.register("transformers_clf")(ClassifierModelProvider)

    # OrcaRouter Cyber Inference (https://www.orcarouter.ai/cyber)
    from circleguardbench.providers.orcarouter_providers import (
        register_orcarouter_providers,
    )

    register_orcarouter_providers(MODEL_PROVIDERS_REGISTRY)
