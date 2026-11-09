#!/usr/bin/env python3

import sys
import logging
from typing import Optional

import click

from circleguardbench.context import GuardbenchContext
from circleguardbench.evaluator import Evaluator
from tqdm import tqdm

from circleguardbench.models_config import ModelType
from circleguardbench.orcarouter.cli import orcarouter as orcarouter_group


logger = logging.getLogger("guardbench.cli")

# Create context instance
pass_context = click.make_pass_decorator(GuardbenchContext, ensure=True)


def root_context() -> GuardbenchContext:
    """Create the shared GuardbenchContext for the root group's click ctx.

    The context is stored on the raw click ctx so subcommands (whose own
    ``pass_context`` wraps it) can reach it via :func:`get_context`."""
    click_ctx = click.get_current_context()
    gctx = GuardbenchContext()
    click_ctx.obj = gctx
    return gctx


def get_context(cli_ctx: click.Context) -> GuardbenchContext:
    """`orcarouter` subcommands receive a plain click ctx; the shared
    GuardbenchContext lives on the root group (created by `pass_context`)."""
    return cli_ctx.find_root().obj


@click.group()
@click.option(
    "--bench_data",
    "-bd",
    default="whitecircle-ai/circleguardbench_public",
    help="Path to the dataset used for evaluation, must follow GuardBench format",
)
@click.option(
    "--models",
    "-m",
    default="configs/models.json",
    help="Path to the models configuration file",
)
@click.option(
    "--skip-models-load",
    is_flag=True,
    help="Skip loading models configuration (for cached evaluation only)",
)
@click.option(
    "--prompts",
    "-p",
    default="prompts",
    help="Path to the directory containing prompt templates",
)
@click.option(
    "--results_dir", default="results", help="Directory to store benchmark results"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def cli(
    bench_data: str,
    models: str,
    skip_models_load: bool,
    prompts: str,
    results_dir: str,
    verbose: bool,
):
    """Guardbench CLI for neural network model evaluation."""
    if verbose:
        logging.getLogger("guardbench").setLevel(logging.DEBUG)

    # `orcarouter` only needs prompt templates and a models file, not the
    # benchmark dataset (see `orcarouter` commands below); the dataset is
    # loaded lazily by commands that actually evaluate.
    click_ctx = click.get_current_context()
    is_orcarouter = click_ctx.invoked_subcommand == "orcarouter"

    gctx = root_context()

    # Load models configuration
    if not skip_models_load:
        try:
            gctx.load_models(models)
        except Exception as e:
            logger.critical(f"Failed to load models configuration: {e}")
            click.echo(f"Critical error: {e}", err=True)
            sys.exit(1)

    # Load prompt templates
    try:
        gctx.load_prompt_templates(prompts)
    except Exception as e:
        logger.critical(f"Failed to load prompt templates: {e}")
        click.echo(f"Critical error: {e}", err=True)
        sys.exit(1)

    if not is_orcarouter:
        # Load dataset
        try:
            gctx.load_dataset(bench_data)
        except Exception as e:
            logger.critical(f"Failed to load dataset: {e}")
            click.echo(f"Critical error: {e}", err=True)
            sys.exit(1)

    gctx.results_dir = results_dir
    gctx.is_initialized = True


@cli.command()
@click.argument("model_name", required=False)
@click.option("--all", is_flag=True, help="Run evaluation for all models")
@click.option(
    "--force", is_flag=True, help="Force evaluation even for already evaluated models"
)
@click.option(
    "--model-type",
    type=click.Choice([t.value for t in ModelType]),
    help="Model type for cached evaluation (required when using cached results)",
)
@pass_context
def run(
    ctx: GuardbenchContext,
    model_name: Optional[str],
    all: bool,
    force: bool,
    model_type: Optional[str],
):
    """
    Run evaluation for a specific model or all models.

    If the --all flag is specified, all models from the configuration will be evaluated.
    Otherwise, a specific model name must be provided.

    When using --skip-models-load, you must provide --model-type to update the leaderboard
    with cached results.
    """
    try:
        using_cached = ctx.models_config is None

        evaluator = Evaluator(ctx, force, using_cached)

        if using_cached:
            if all:
                raise ValueError("Cannot use --all with --skip-models-load")
            if not model_name:
                raise ValueError("Model name is required when using --skip-models-load")
            if not model_type:
                raise ValueError(
                    "--model-type is required when using --skip-models-load"
                )

            logger.info(
                f"Calculating leaderboard for model {model_name} from cached results"
            )

            evaluator.evaluate_model(model_name, model_type)

            logger.info(
                f"Leaderboard updated for model {model_name} from cached results"
            )
        else:
            if all:
                logger.info("Starting evaluation for all models...")

                model_list = ctx.models_config.list()
                for model in tqdm(model_list, desc="Evaluating models"):
                    logger.info(f"Starting evaluation for model: {model}...")
                    evaluator.evaluate_model(model)

                logger.info("Evaluation of all models completed!")
            elif model_name:
                if model_name not in ctx.models_config:
                    raise ValueError(f"Model '{model_name}' not found in configuration")
                logger.info(f"Starting evaluation for model: {model_name}...")
                evaluator.evaluate_model(model_name)
                logger.info(f"Evaluation of model {model_name} completed!")
            else:
                click.echo(
                    "Please specify a model name or use the --all flag", err=True
                )
                sys.exit(1)
    except ValueError as e:
        logger.error(str(e))
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred")
        click.echo(f"Error during evaluation: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--use-categories/--no-categories",
    default=False,
    help="Group results by categories",
)
@click.option(
    "--sort-by",
    "-s",
    type=click.Choice(
        [
            "accuracy",
            "recall",
            "error_ratio",
            "avg_runtime_ms",
        ]
    ),
    default="accuracy",
    help="Field to sort by",
)
@click.option(
    "--metric-type",
    "-m",
    type=click.Choice(
        [
            "default_prompts",
            "jailbreaked_prompts",
            "default_answers",
            "jailbreaked_answers",
        ]
    ),
    default=None,
    help="Metric type to display (if not specified, all are shown)",
)
@pass_context
def leaderboard(
    ctx: GuardbenchContext,
    use_categories: bool,
    sort_by: str,
    metric_type: Optional[str],
):
    """
    Display the leaderboard with model evaluation results.

    Results are grouped by categories by default.
    Use --no-categories to disable grouping.
    """
    try:
        from circleguardbench.leaderboard import Leaderboard

        leaderboard_instance = Leaderboard(results_dir=ctx.results_dir)
        click.echo("Loading leaderboard...")
        leaderboard_instance.show_leaderboard(
            bench_name=ctx.bench_name,
            use_categories=use_categories,
            sort_by=sort_by,
            metric_type=metric_type,
        )
    except Exception as e:
        logger.exception("Error displaying leaderboard")
        click.echo(f"Error displaying leaderboard: {e}", err=True)
        sys.exit(1)


@cli.command()
@pass_context
def models(ctx: GuardbenchContext):
    """Display the models configuration."""
    click.echo("Models Configuration:")
    for model_name in ctx.models_config.list():
        model_config = ctx.models_config[model_name]
        click.echo(f"\n{model_name}:")
        click.echo(f"  Type: {model_config.type}")
        click.echo(f"  Evaluation on: {model_config.eval_on}")
        click.echo(f"  Inference engine: {model_config.inference_engine}")
        click.echo(f"  Max concurrency: {model_config.max_concurrency}")

        if model_config.use_cot:
            click.echo(f"  Use CoT: {model_config.use_cot}")

        if model_config.params:
            click.echo("  Parameters:")
            if hasattr(model_config.params, "model_dump"):
                params_dict = model_config.params.model_dump()
            else:
                params_dict = model_config.params

            for key, value in params_dict.items():
                if "key" in key.lower() and value:
                    value = "********"
                click.echo(f"    {key}: {value}")


@cli.command()
@pass_context
def prompts(ctx: GuardbenchContext):
    """Display the available prompt templates."""
    click.echo("Available Prompt Templates:")
    for template_name in sorted(ctx.prompts_templates.keys()):
        click.echo(f"- {template_name}")


@cli.command()
@pass_context
def dataset_info(ctx: GuardbenchContext):
    """Display information about the loaded dataset."""
    if not ctx.dataset:
        click.echo("Dataset not loaded")
        return

    click.echo("Dataset Information:")
    click.echo(f"Source: {ctx.bench_name}")

    # Work with dataset directly, without splits
    click.echo(f"Number of examples: {len(ctx.dataset)}")
    click.echo(f"Columns: {', '.join(ctx.dataset.column_names)}")

    click.echo(f"Categories ({len(ctx.dataset_categories)}):")
    for category in sorted(ctx.dataset_categories):
        count = sum(1 for c in ctx.dataset["harm_category"] if c == category)
        click.echo(f"  - {category}: {count} examples")


cli.add_command(orcarouter_group, "orcarouter")


if __name__ == "__main__":
    cli()
