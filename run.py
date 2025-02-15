import os
import warnings

import dotenv
import hydra
import rich
from omegaconf import DictConfig, OmegaConf
from rich.traceback import install
from rich.tree import Tree
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from gate.base.utils.loggers import get_logger
from gate.base.utils.rank_zero_ops import extras

# load environment variables from `.env-` file if it exists
# recursively searches for `.env` in all folders starting from work dir

dotenv.load_dotenv(override=True, verbose=True)
install(show_locals=False, word_wrap=True, width=350)
log = get_logger(__name__)

from gate.configs.config import collect_config_store
from gate.configs.config import collect_config_store

config_store = collect_config_store()


def print_config_store_options(config_store: DictConfig):
    style = "dim"
    tree = Tree("CONFIG", style=style, guide_style=style)

    for key, value in config_store.repo.items():
        branch = tree.add(key, style=style, guide_style=style)

        branch_content = str(value)
        if isinstance(value, DictConfig):
            branch_content = OmegaConf.to_yaml(dict(value))

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_store_tree.log", "w") as fp:
        rich.print(tree, file=fp)


@hydra.main(version_base=None, config_name="config")
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    print_config_store_options(config_store)

    from gate.train_eval import train_eval

    os.environ["WANDB_PROGRAM"] = config.code_dir

    extras(config)

    return train_eval(config)


if __name__ == "__main__":
    main()
