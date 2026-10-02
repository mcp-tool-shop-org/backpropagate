"""Plain-language explanations for the web UI's "i" tips.

One place for every explanation, so the words can be read, reviewed and
tested as a set. Each tip answers three things a curious newcomer asks:
what is this, what happens if I change it, and where should I start.

House rules for the copy:

* Plain words first, the technical term second, never jargon alone.
* Say what a setting DOES to the result or to memory, not only what it is.
* A starting point where one exists. No "it depends" without the dependence.
* Nothing here is a promise about quality: these are the common defaults and
  the reasons behind them.

Tips are looked up by key (``TIPS["rank"]``). ``tests/test_ui_help_text.py``
checks that every key used by a page exists and that the copy stays short.
"""

from __future__ import annotations

from dataclasses import dataclass

HANDBOOK = "https://mcp-tool-shop-org.github.io/backpropagate/handbook"


@dataclass(frozen=True)
class Tip:
    """One explanation: a title, one to three short paragraphs, and an
    optional starting point and handbook page."""

    title: str
    body: tuple[str, ...]
    start: str = ""
    link: str = ""  # a handbook path such as "/training/"

    @property
    def text(self) -> str:
        """Everything as one string (the screen-reader description)."""
        parts = [self.title + ".", *self.body]
        if self.start:
            parts.append("Good starting point: " + self.start)
        return " ".join(parts)

    @property
    def url(self) -> str:
        return HANDBOOK + self.link if self.link else ""


TIPS: dict[str, Tip] = {
    # ---- pages ---------------------------------------------------------------
    "page_single_run": Tip(
        "What fine-tuning does",
        (
            "Fine-tuning teaches an existing model from your examples. You show it "
            "the kind of answers you want, and it adjusts itself to answer that way.",
            "It runs on your own graphics card and your examples stay on this "
            "computer. The model itself is downloaded once.",
            "When it finishes you have a small add-on file, called an adapter. You "
            "can test it, export it to Ollama, or train it further.",
        ),
        start="Pick a model, point to a dataset, press Start. The defaults follow your GPU.",
        link="/beginners/",
    ),
    "page_multi_run": Tip(
        "Why several short runs",
        (
            "A multi-run trains in several short rounds instead of one long one. "
            "After each round the new adapter is merged with the previous ones.",
            "Training on new data can make a model forget what it learned earlier. "
            "Merging between rounds keeps the earlier learning while adding the new.",
            "Use it for larger datasets or longer training. For a first try, a "
            "single run is simpler.",
        ),
        link="/training/#multi-run-slao-training",
    ),
    "runs": Tip(
        "How many rounds",
        (
            "Each run is a short round of training on a fresh part of your "
            "dataset. After each one, its adapter is merged into the result so far.",
        ),
        start="3.",
    ),
    "samples_per_run": Tip(
        "Examples used in each round",
        (
            "How many examples each run takes from your dataset. Runs times "
            "samples is how much of the dataset the whole multi-run uses.",
        ),
        start="Enough that runs times samples covers your dataset once.",
    ),
    "merge_mode": Tip(
        "How the rounds are combined",
        (
            "SLAO merges each new adapter in a way that keeps the earlier "
            "learning. It is made for this and is the default.",
            "Simple average weighs every run equally. TIES keeps the strongest "
            "changes from each run and drops the ones that conflict.",
        ),
        start="SLAO.",
    ),
    # ---- model ---------------------------------------------------------------
    "model": Tip(
        "The model you start from",
        (
            "A base model has already been trained by someone else. It knows "
            "language; fine-tuning only teaches it your task.",
            "Bigger models are more capable and need more GPU memory. The number "
            "in a name such as 7B is its size in billions of parameters.",
            "A preset is a model we have tried. You can also type any model id "
            "from huggingface.co, or the path of a model folder on this computer.",
        ),
        start="The estimate at the bottom of the page tells you whether a model fits your GPU.",
        link="/training/#model-presets",
    ),
    # ---- dataset -------------------------------------------------------------
    "dataset": Tip(
        "Your examples",
        (
            "A dataset is a file of examples: a question or instruction, and the "
            "answer you want. One example per line, in a .jsonl file.",
            "The common chat layouts are recognised automatically: ShareGPT, "
            "Alpaca and OpenAI messages.",
            "Clear, consistent examples matter more than a large number of them. "
            "The Dataset page shows you what a file contains before you train.",
        ),
        start="A few hundred good examples are enough for a first run.",
        link="/training/#dataset-formats",
    ),
    # ---- method --------------------------------------------------------------
    "method": Tip(
        "How the model learns from your data",
        (
            "SFT learns from examples: each one shows a good answer. This is the "
            "standard choice.",
            "ORPO and SimPO learn from preferences: each example has a better and "
            "a worse answer, and the model learns to prefer the better one.",
            "KTO learns from single answers marked good or bad, like thumbs-up and "
            "thumbs-down.",
        ),
        start="SFT, unless your data has chosen and rejected answers.",
        link="/preference-tuning/",
    ),
    "method_knobs": Tip(
        "Preference settings",
        (
            "These control how strongly the model is pushed towards the preferred "
            "answer and away from the rejected one.",
            "The values shown are the defaults the trainer uses.",
        ),
        start="Leave them as they are for a first run.",
    ),
    # ---- mode ----------------------------------------------------------------
    "mode": Tip(
        "How much of the model is trained",
        (
            "QLoRA compresses the base model to 4 bits, about a third of the "
            "memory, and trains a small adapter on top. It needs the least memory "
            "and the result is close for most tasks.",
            "LoRA trains the same adapter on the uncompressed model. It needs "
            "about three times more memory for the model.",
            "Full fine-tune changes every weight of the model. It needs the most "
            "memory, so it suits small models.",
        ),
        start="QLoRA.",
        link="/full-fine-tuning/",
    ),
    # ---- training shape ------------------------------------------------------
    "steps": Tip(
        "How long to train",
        (
            "One step is one update of the model, made from one batch of examples. "
            "More steps mean more learning and more time.",
            "Too few and the model barely changes. Too many on a small dataset and "
            "it starts memorising the examples instead of learning from them.",
        ),
        start="100 for a quick trial. When the loss curve flattens, more steps add little.",
    ),
    "batch_size": Tip(
        "Examples per step",
        (
            "How many examples the model looks at for each update. A bigger batch "
            "gives steadier updates and uses more GPU memory.",
            "On auto, a batch is chosen that fits your GPU for the model and LoRA "
            "shape you picked.",
        ),
        start="Leave it on auto.",
    ),
    "learning_rate": Tip(
        "How big each update is",
        (
            "Too high and training becomes unstable: the loss jumps around or "
            "climbs. Too low and the model learns slowly.",
            "0.0002 is the usual value for LoRA training.",
        ),
        start="0.0002. Halve it if the loss curve is erratic.",
    ),
    # ---- LoRA ----------------------------------------------------------------
    "lora": Tip(
        "The adapter and its size",
        (
            "LoRA trains a small add-on, called an adapter, instead of changing the "
            "whole model. The adapter is what you keep at the end.",
            "The shape decides how much the adapter can learn and how much memory "
            "training needs. Quality is the largest, Balanced is a quarter of it, "
            "and Fast is a small adapter that is quick to try things with.",
            "The recommended shape is the largest one that fits the memory free on "
            "your GPU for this model.",
        ),
        link="/estimate-vram/",
    ),
    "rank": Tip(
        "The size of the adapter",
        (
            "A higher rank lets the adapter capture more of your data. It also "
            "needs more memory and makes a larger file.",
            "16 is small, 64 is medium, 256 is large.",
        ),
        start="Use a shape above; change the rank when you know you need to.",
    ),
    "alpha": Tip(
        "How strongly the adapter is applied",
        (
            "Alpha scales the adapter's effect on the model. The convention is "
            "twice the rank, and the shapes above follow it.",
        ),
        start="Twice the rank.",
    ),
    "dropout": Tip(
        "A guard against memorising",
        (
            "During training a small share of the adapter is switched off at "
            "random, so the model cannot lean on memorised examples. 0.05 means 5%.",
        ),
        start="0.05. Up to 0.1 for a very small dataset.",
    ),
    "target_modules": Tip(
        "Which parts of the model get an adapter",
        (
            "all-linear puts an adapter on every layer type. It learns the most "
            "and uses the most memory.",
            "q_proj, v_proj covers only two attention layers in each block. The "
            "adapter is much smaller and trains faster.",
        ),
        start="all-linear.",
    ),
    # ---- advanced ------------------------------------------------------------
    "gpu_temp": Tip(
        "A safety stop for heat",
        (
            "If the GPU stays at or above this temperature, the run saves a "
            "checkpoint and stops, as if you had pressed Stop. You can continue "
            "from the checkpoint later.",
        ),
        start="90 °C.",
    ),
    "run_name": Tip(
        "A label for this run",
        (
            "The name this run gets in an experiment tracker such as Weights & "
            "Biases, TensorBoard or MLflow, if you use one. It changes nothing "
            "about the training.",
        ),
    ),
    "gradient_checkpointing": Tip(
        "Less memory for a little time",
        (
            "The model recomputes some of its intermediate results instead of "
            "keeping them all in memory. Training is slightly slower and needs "
            "several times less memory for each example.",
        ),
        start="On, unless you have memory to spare.",
    ),
    # ---- the estimate --------------------------------------------------------
    "vram": Tip(
        "Will it fit on your GPU?",
        (
            "VRAM is your graphics card's memory. Training has to fit in it. When "
            "it does not, the run fails or the whole computer slows down.",
            "Fits is comfortable. Tight may work but is close to the limit. Won't "
            "fit needs a smaller batch, a smaller LoRA shape or a smaller model.",
            "The number assumes every example is as long as the limit of 2,048 "
            "tokens. Shorter examples use less.",
        ),
        link="/estimate-vram/",
    ),
    "measure": Tip(
        "Replace the estimate with a measurement",
        (
            "Runs a few seconds of real training with this model on your GPU and "
            "records what it needs. It takes a minute or two.",
            "It is careful with your machine: it only runs steps that are expected "
            "to fit, and it stops cleanly if memory runs short.",
        ),
        link="/estimate-vram/#measure-on-your-own-gpu",
    ),
    # ---- the side panel ------------------------------------------------------
    "run_state": Tip(
        "What is happening now",
        (
            "Idle means nothing is running. During a run this shows the step, the "
            "speed and the time left.",
            "You can leave this page or reload it; the run keeps going and the "
            "panel picks it up again.",
        ),
    ),
    "loss": Tip(
        "How wrong the model still is",
        (
            "Loss measures how far the model's answers are from your examples. "
            "Lower is better.",
            "A healthy curve falls quickly, then levels off. A curve that jumps "
            "around or climbs usually means the learning rate is too high.",
        ),
    ),
    "gpu": Tip(
        "Your graphics card right now",
        (
            "The temperature and the memory in use by everything on the card, not "
            "only by training.",
            "If memory is already well used before you start, close other programs "
            "that use the GPU.",
        ),
    ),
    "events": Tip(
        "A short log of the run",
        (
            "The latest messages from the run: loading, training, saving, and any "
            "warning. The full log has every line.",
        ),
    ),
}


def tip(key: str) -> Tip:
    """The tip for ``key``. A missing key is a programming error."""
    return TIPS[key]


__all__ = ["HANDBOOK", "TIPS", "Tip", "tip"]
