import json
import os
from dataclasses import dataclass, field

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from transformers import AutoTokenizer, BertForMaskedLM, CLIPTextModel

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import barrier, cleanup, get_rank
from threestudio.utils.ops import shifted_cosine_decay, shifted_expotional_decay
from threestudio.utils.typing import *

import safetensors
from diffusers.utils import _get_model_file

def hash_prompt(model: str, prompt: str) -> str:
    import hashlib

    identifier = f"{model}-{prompt}"
    return hashlib.md5(identifier.encode()).hexdigest()


@dataclass
class DirectionConfig:
    name: str
    prompt: Callable[[str], str]
    negative_prompt: Callable[[str], str]
    condition: Callable[
        [Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]],
        Float[Tensor, "B"],
    ]


@dataclass
class PromptProcessorOutput:
    base_text_embeddings: Float[Tensor, "N Nf"]
    src_text_embeddings: Float[Tensor, "N Nf"]
    tgt_text_embeddings: Float[Tensor, "N Nf"]
    uncond_text_embeddings: Float[Tensor, "N Nf"]
    base_text_embeddings_vd: Float[Tensor, "Nv N Nf"]
    src_text_embeddings_vd: Float[Tensor, "Nv N Nf"]
    tgt_text_embeddings_vd: Float[Tensor, "Nv N Nf"]
    uncond_text_embeddings_vd: Float[Tensor, "Nv N Nf"]
    directions: List[DirectionConfig]
    direction2idx: Dict[str, int]
    use_perp_neg: bool
    perp_neg_f_sb: Tuple[float, float, float]
    perp_neg_f_fsb: Tuple[float, float, float]
    perp_neg_f_fs: Tuple[float, float, float]
    perp_neg_f_sf: Tuple[float, float, float]
    prompt: str
    base_prompts_vd: List[str]
    src_prompts_vd: List[str]
    tgt_prompts_vd: List[str]

    def get_text_embeddings(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        view_dependent_prompting: bool = True,
    ) -> Float[Tensor, "BB N Nf"]:
        batch_size = elevation.shape[0]

        if view_dependent_prompting:
            # Get direction
            direction_idx = torch.zeros_like(elevation, dtype=torch.long)
            for d in self.directions:
                direction_idx[
                    d.condition(elevation, azimuth, camera_distances)
                ] = self.direction2idx[d.name]

            # Get text embeddings
            base_text_embeddings = self.base_text_embeddings_vd[direction_idx]  # type: ignore
            src_text_embeddings = self.src_text_embeddings_vd[direction_idx]  # type: ignore
            tgt_text_embeddings = self.tgt_text_embeddings_vd[direction_idx]  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings_vd[direction_idx]  # type: ignore
        else:
            base_text_embeddings = self.base_text_embeddings.expand(batch_size, -1, -1)  # type: ignore
            src_text_embeddings = self.src_text_embeddings.expand(batch_size, -1, -1)  # type: ignore
            tgt_text_embeddings = self.tgt_text_embeddings.expand(batch_size, -1, -1)  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings.expand(  # type: ignore
                batch_size, -1, -1
            )

        # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        return torch.cat([src_text_embeddings, uncond_text_embeddings], dim=0), torch.cat([tgt_text_embeddings, uncond_text_embeddings], dim=0), torch.cat([base_text_embeddings, uncond_text_embeddings], dim=0)

    def get_text_embeddings_perp_neg(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        view_dependent_prompting: bool = True,
    ) -> Tuple[Float[Tensor, "BBBB N Nf"], Float[Tensor, "B 2"]]:
        assert (
            view_dependent_prompting
        ), "Perp-Neg only works with view-dependent prompting"

        batch_size = elevation.shape[0]

        direction_idx = torch.zeros_like(elevation, dtype=torch.long)
        for d in self.directions:
            direction_idx[
                d.condition(elevation, azimuth, camera_distances)
            ] = self.direction2idx[d.name]
        # 0 - side view
        # 1 - front view
        # 2 - back view
        # 3 - overhead view

        pos_text_embeddings = []
        neg_text_embeddings = []
        neg_guidance_weights = []
        uncond_text_embeddings = []

        side_emb = self.text_embeddings_vd[0]
        front_emb = self.text_embeddings_vd[1]
        back_emb = self.text_embeddings_vd[2]
        overhead_emb = self.text_embeddings_vd[3]

        for idx, ele, azi, dis in zip(
            direction_idx, elevation, azimuth, camera_distances
        ):
            azi = shift_azimuth_deg(azi)  # to (-180, 180)
            uncond_text_embeddings.append(
                self.uncond_text_embeddings_vd[idx]
            )  # should be ""
            if idx.item() == 3:  # overhead view
                pos_text_embeddings.append(overhead_emb)  # side view
                # dummy
                neg_text_embeddings += [
                    self.uncond_text_embeddings_vd[idx],
                    self.uncond_text_embeddings_vd[idx],
                ]
                neg_guidance_weights += [0.0, 0.0]
            else:  # interpolating views
                if torch.abs(azi) < 90:
                    # front-side interpolation
                    # 0 - complete side, 1 - complete front
                    r_inter = 1 - torch.abs(azi) / 90
                    pos_text_embeddings.append(
                        r_inter * front_emb + (1 - r_inter) * side_emb
                    )
                    neg_text_embeddings += [front_emb, side_emb]
                    neg_guidance_weights += [
                        -shifted_expotional_decay(*self.perp_neg_f_fs, r_inter),
                        -shifted_expotional_decay(*self.perp_neg_f_sf, 1 - r_inter),
                    ]
                else:
                    # side-back interpolation
                    # 0 - complete back, 1 - complete side
                    r_inter = 2.0 - torch.abs(azi) / 90
                    pos_text_embeddings.append(
                        r_inter * side_emb + (1 - r_inter) * back_emb
                    )
                    neg_text_embeddings += [side_emb, front_emb]
                    neg_guidance_weights += [
                        -shifted_expotional_decay(*self.perp_neg_f_sb, r_inter),
                        -shifted_expotional_decay(*self.perp_neg_f_fsb, r_inter),
                    ]

        text_embeddings = torch.cat(
            [
                torch.stack(pos_text_embeddings, dim=0),
                torch.stack(uncond_text_embeddings, dim=0),
                torch.stack(neg_text_embeddings, dim=0),
            ],
            dim=0,
        )

        return text_embeddings, torch.as_tensor(
            neg_guidance_weights, device=elevation.device
        ).reshape(batch_size, 2)


def shift_azimuth_deg(azimuth: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    # shift azimuth angle (in degrees), to [-180, 180]
    return (azimuth + 180) % 360 - 180


class PromptProcessor(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        prompt: str = "a hamburger"
        src_modifier: str = 'oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed'
        tgt_modifier: str = '.'
        texture_inversion_embedding: str = ''

        # manually assigned view-dependent prompts
        src_prompt_front: Optional[str] = None
        src_prompt_side: Optional[str] = None
        src_prompt_back: Optional[str] = None
        src_prompt_overhead: Optional[str] = None
        tgt_prompt_front: Optional[str] = None
        tgt_prompt_side: Optional[str] = None
        tgt_prompt_back: Optional[str] = None
        tgt_prompt_overhead: Optional[str] = None

        negative_prompt: str = ""
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        overhead_threshold: float = 60.0
        front_threshold: float = 45.0
        back_threshold: float = 45.0
        view_dependent_prompt_front: bool = False
        use_cache: bool = True
        spawn: bool = True

        # perp neg
        use_perp_neg: bool = False
        # a*e(-b*r) + c
        # a * e(-b) + c = 0
        perp_neg_f_sb: Tuple[float, float, float] = (1, 0.5, -0.606)
        perp_neg_f_fsb: Tuple[float, float, float] = (1, 0.5, +0.967)
        perp_neg_f_fs: Tuple[float, float, float] = (
            4,
            0.5,
            -2.426,
        )  # f_fs(1) = 0, a, b > 0
        perp_neg_f_sf: Tuple[float, float, float] = (4, 0.5, -2.426)

        # prompt debiasing
        use_prompt_debiasing: bool = False
        pretrained_model_name_or_path_prompt_debiasing: str = "bert-base-uncased"
        # index of words that can potentially be removed
        prompt_debiasing_mask_ids: Optional[List[int]] = None

        use_modifier_only: bool = True

    cfg: Config

    @rank_zero_only
    def configure_text_encoder(self) -> None:
        raise NotImplementedError

    @rank_zero_only
    def destroy_text_encoder(self) -> None:
        raise NotImplementedError

    def configure(self) -> None:
        self._cache_dir = ".threestudio_cache/text_embeddings"  # FIXME: hard-coded path

        # view-dependent text embeddings
        self.directions: List[DirectionConfig]
        if self.cfg.view_dependent_prompt_front:
            self.directions = [
                DirectionConfig(
                    "side",
                    lambda s: f"side view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool),
                ),
                DirectionConfig(
                    "front",
                    lambda s: f"front view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > -self.cfg.front_threshold
                    )
                    & (shift_azimuth_deg(azi) < self.cfg.front_threshold),
                ),
                DirectionConfig(
                    "back",
                    lambda s: f"backside view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > 180 - self.cfg.back_threshold
                    )
                    | (shift_azimuth_deg(azi) < -180 + self.cfg.back_threshold),
                ),
                DirectionConfig(
                    "overhead",
                    lambda s: f"overhead view of {s}",
                    lambda s: s,
                    lambda ele, azi, dis: ele > self.cfg.overhead_threshold,
                ),
            ]
        else:
            self.directions = [
                DirectionConfig(
                    "side",
                    lambda s: f"{s}, side view",
                    lambda s: s,
                    lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool),
                ),
                DirectionConfig(
                    "front",
                    lambda s: f"{s}, front view",
                    lambda s: s,
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > -self.cfg.front_threshold
                    )
                    & (shift_azimuth_deg(azi) < self.cfg.front_threshold),
                ),
                DirectionConfig(
                    "back",
                    lambda s: f"{s}, back view",
                    lambda s: s,
                    lambda ele, azi, dis: (
                        shift_azimuth_deg(azi) > 180 - self.cfg.back_threshold
                    )
                    | (shift_azimuth_deg(azi) < -180 + self.cfg.back_threshold),
                ),
                DirectionConfig(
                    "overhead",
                    lambda s: f"{s}, overhead view",
                    lambda s: s,
                    lambda ele, azi, dis: ele > self.cfg.overhead_threshold,
                ),
            ]

        self.direction2idx = {d.name: i for i, d in enumerate(self.directions)}

        if os.path.exists("load/prompt_library.json"):
            with open(os.path.join("load/prompt_library.json"), "r") as f:
                self.prompt_library = json.load(f)
        else:
            self.prompt_library = {}
        # use provided prompt or find prompt in library
        self.prompt = self.preprocess_prompt(self.cfg.prompt)
        # use provided negative prompt
        self.negative_prompt = self.cfg.negative_prompt

        # process sds bridge source and target prompt
        if self.cfg.use_modifier_only:
            self.src_prompt = self.cfg.src_modifier
        else:
            self.src_prompt = self.prompt + ', ' + self.cfg.src_modifier

        self.tgt_prompt = self.prompt + ', ' + self.cfg.tgt_modifier

        threestudio.info(
            f"Using prompt [{self.prompt}] and negative prompt [{self.negative_prompt}]"
        )

        # view-dependent prompting
        if self.cfg.use_prompt_debiasing:
            # Warning: not implemented with sds bridge yet
            assert (
                self.cfg.prompt_side is None
                and self.cfg.prompt_back is None
                and self.cfg.prompt_overhead is None
            ), "Do not manually assign prompt_side, prompt_back or prompt_overhead when using prompt debiasing"
            prompts = self.get_debiased_prompt(self.prompt)
            self.prompts_vd = [
                d.prompt(prompt) for d, prompt in zip(self.directions, prompts)
            ]
        else:
            self.base_prompts_vd = [
                self.cfg.get(f"src_prompt_{d.name}", None) or d.prompt(self.prompt)  # type: ignore
                for d in self.directions
            ]
            self.src_prompts_vd = [
                self.cfg.get(f"src_prompt_{d.name}", None) or d.prompt(self.src_prompt)  # type: ignore
                for d in self.directions
            ]
            self.tgt_prompts_vd = [
                self.cfg.get(f"tgt_prompt_{d.name}", None) or d.prompt(self.tgt_prompt)  # type: ignore
                for d in self.directions
            ]

        prompts_vd_display = " ".join(
            [
                f"[{d.name}]:[{prompt}]"
                for prompt, d in zip(self.src_prompts_vd, self.directions)
            ]
        )
        threestudio.info(f"Using source view-dependent prompts {prompts_vd_display}")

        prompts_vd_display = " ".join(
            [
                f"[{d.name}]:[{prompt}]"
                for prompt, d in zip(self.tgt_prompts_vd, self.directions)
            ]
        )
        threestudio.info(f"Using target view-dependent prompts {prompts_vd_display}")

        self.negative_prompts_vd = [
            d.negative_prompt(self.negative_prompt) for d in self.directions
        ]

        self.prepare_text_embeddings()
        self.load_text_embeddings()

    def spawn_func(self, pretrained_model_name_or_path, prompts, cache_dir):
        raise NotImplementedError

    @rank_zero_only
    def prepare_text_embeddings(self):
        os.makedirs(self._cache_dir, exist_ok=True)

        all_prompts = (
            [self.src_prompt, self.tgt_prompt, self.prompt]
            + [self.negative_prompt]
            + self.src_prompts_vd
            + self.tgt_prompts_vd
            + self.base_prompts_vd
            + self.negative_prompts_vd
        )
        prompts_to_process = []
        for prompt in all_prompts:
            if self.cfg.use_cache and '<sds bridge>' not in prompt:
                # some text embeddings are already in cache
                # do not process them
                cache_path = os.path.join(
                    self._cache_dir,
                    f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
                )
                if os.path.exists(cache_path):
                    threestudio.debug(
                        f"Text embeddings for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] are already in cache, skip processing."
                    )
                    continue
            prompts_to_process.append(prompt)

        if len(prompts_to_process) > 0:
            if False: # deprecated for now
                ctx = mp.get_context("spawn")
                subprocess = ctx.Process(
                    target=self.spawn_func,
                    args=(
                        self.cfg.pretrained_model_name_or_path,
                        prompts_to_process,
                        self._cache_dir,
                    ),
                )
                subprocess.start()
                subprocess.join()
                assert subprocess.exitcode == 0, "prompt embedding process failed!"
            else:
                self.spawn_func(
                    self.cfg.pretrained_model_name_or_path,
                    prompts_to_process,
                    self._cache_dir,
                )
            cleanup()

    def load_text_embeddings(self):
        # synchronize, to ensure the text embeddings have been computed and saved to cache
        barrier()
        self.base_text_embeddings = self.load_from_cache(self.prompt)[None, ...]
        self.src_text_embeddings = self.load_from_cache(self.src_prompt)[None, ...]
        self.tgt_text_embeddings = self.load_from_cache(self.tgt_prompt)[None, ...]
        self.uncond_text_embeddings = self.load_from_cache(self.negative_prompt)[
            None, ...
        ]
        self.base_text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt) for prompt in self.base_prompts_vd], dim=0
        )
        self.src_text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt) for prompt in self.src_prompts_vd], dim=0
        )
        self.tgt_text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt) for prompt in self.tgt_prompts_vd], dim=0
        )
        self.uncond_text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt) for prompt in self.negative_prompts_vd], dim=0
        )
        threestudio.debug(f"Loaded text embeddings.")

    def load_from_cache(self, prompt):
        cache_path = os.path.join(
            self._cache_dir,
            f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Text embedding file {cache_path} for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] not found."
            )
        return torch.load(cache_path, map_location=self.device)

    def preprocess_prompt(self, prompt: str) -> str:
        if prompt.startswith("lib:"):
            # find matches in the library
            candidate = None
            keywords = prompt[4:].lower().split("_")
            for prompt in self.prompt_library["dreamfusion"]:
                if all([k in prompt.lower() for k in keywords]):
                    if candidate is not None:
                        raise ValueError(
                            f"Multiple prompts matched with keywords {keywords} in library"
                        )
                    candidate = prompt
            if candidate is None:
                raise ValueError(
                    f"Cannot find prompt with keywords {keywords} in library"
                )
            threestudio.info("Find matched prompt in library: " + candidate)
            return candidate
        else:
            return prompt

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B ..."], Float[Tensor, "B ..."]]:
        raise NotImplementedError

    def get_debiased_prompt(self, prompt: str) -> List[str]:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path_prompt_debiasing
        )
        model = BertForMaskedLM.from_pretrained(
            self.cfg.pretrained_model_name_or_path_prompt_debiasing
        )

        views = [d.name for d in self.directions]
        view_ids = tokenizer(" ".join(views), return_tensors="pt").input_ids[0]
        view_ids = view_ids[1:5]

        def modulate(prompt):
            prompt_vd = f"This image is depicting a [MASK] view of {prompt}"
            tokens = tokenizer(
                prompt_vd,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            mask_idx = torch.where(tokens.input_ids == tokenizer.mask_token_id)[1]

            logits = model(**tokens).logits
            logits = F.softmax(logits[0, mask_idx], dim=-1)
            logits = logits[0, view_ids]
            probes = logits / logits.sum()
            return probes

        prompts = [prompt.split(" ") for _ in range(4)]
        full_probe = modulate(prompt)
        n_words = len(prompt.split(" "))
        prompt_debiasing_mask_ids = (
            self.cfg.prompt_debiasing_mask_ids
            if self.cfg.prompt_debiasing_mask_ids is not None
            else list(range(n_words))
        )
        words_to_debias = [prompt.split(" ")[idx] for idx in prompt_debiasing_mask_ids]
        threestudio.info(f"Words that can potentially be removed: {words_to_debias}")
        for idx in prompt_debiasing_mask_ids:
            words = prompt.split(" ")
            prompt_ = " ".join(words[:idx] + words[(idx + 1) :])
            part_probe = modulate(prompt_)

            pmi = full_probe / torch.lerp(part_probe, full_probe, 0.5)
            for i in range(pmi.shape[0]):
                if pmi[i].item() < 0.95:
                    prompts[i][idx] = ""

        debiased_prompts = [" ".join([word for word in p if word]) for p in prompts]
        for d, debiased_prompt in zip(views, debiased_prompts):
            threestudio.info(f"Debiased prompt of the {d} view is [{debiased_prompt}]")

        del tokenizer, model
        cleanup()

        return debiased_prompts

    def __call__(self) -> PromptProcessorOutput:
        return PromptProcessorOutput(
            base_text_embeddings=self.base_text_embeddings,
            src_text_embeddings=self.src_text_embeddings,
            tgt_text_embeddings=self.tgt_text_embeddings,
            uncond_text_embeddings=self.uncond_text_embeddings,
            prompt=self.prompt,
            base_text_embeddings_vd=self.base_text_embeddings_vd,
            src_text_embeddings_vd=self.src_text_embeddings_vd,
            tgt_text_embeddings_vd=self.tgt_text_embeddings_vd,
            uncond_text_embeddings_vd=self.uncond_text_embeddings_vd,
            base_prompts_vd=self.base_prompts_vd,
            src_prompts_vd=self.src_prompts_vd,
            tgt_prompts_vd=self.tgt_prompts_vd,
            directions=self.directions,
            direction2idx=self.direction2idx,
            use_perp_neg=self.cfg.use_perp_neg,
            perp_neg_f_sb=self.cfg.perp_neg_f_sb,
            perp_neg_f_fsb=self.cfg.perp_neg_f_fsb,
            perp_neg_f_fs=self.cfg.perp_neg_f_fs,
            perp_neg_f_sf=self.cfg.perp_neg_f_sf,
        )


    def load_textual_inversion(
        self,
        pretrained_model_name_or_path: Union[str, List[str], Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        text_encoder = None,
        tokenizer = None,
        token: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        user_agent = {
            "file_type": "text_inversion",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path, list):
            pretrained_model_name_or_paths = [pretrained_model_name_or_path]
        else:
            pretrained_model_name_or_paths = pretrained_model_name_or_path

        if isinstance(token, str):
            tokens = [token]
        elif token is None:
            tokens = [None] * len(pretrained_model_name_or_paths)
        else:
            tokens = token

        if len(pretrained_model_name_or_paths) != len(tokens):
            raise ValueError(
                f"You have passed a list of models of length {len(pretrained_model_name_or_paths)}, and list of tokens of length {len(tokens)}"
                f"Make sure both lists have the same length."
            )

        valid_tokens = [t for t in tokens if t is not None]
        if len(set(valid_tokens)) < len(valid_tokens):
            raise ValueError(f"You have passed a list of tokens that contains duplicates: {tokens}")

        token_ids_and_embeddings = []

        for pretrained_model_name_or_path, token in zip(pretrained_model_name_or_paths, tokens):
            if not isinstance(pretrained_model_name_or_path, dict):
                # 1. Load textual inversion file
                model_file = None
                if model_file is None:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=weight_name,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                    )
                    try:
                        state_dict = safetensors.torch.load_file(model_file, device="cpu")
                    except:
                        state_dict = torch.load(model_file, map_location="cpu")
            else:
                state_dict = pretrained_model_name_or_path

            # 2. Load token and embedding correcly from file
            loaded_token = None
            if isinstance(state_dict, torch.Tensor):
                if token is None:
                    raise ValueError(
                        "You are trying to load a textual inversion embedding that has been saved as a PyTorch tensor. Make sure to pass the name of the corresponding token in this case: `token=...`."
                    )
                embedding = state_dict
            elif len(state_dict) == 1:
                # diffusers
                loaded_token, embedding = next(iter(state_dict.items()))
            elif "string_to_param" in state_dict:
                # A1111
                loaded_token = state_dict["name"]
                embedding = state_dict["string_to_param"]["*"]

            if token is not None and loaded_token != token:
                print(f"The loaded token: {loaded_token} is overwritten by the passed token {token}.")
            else:
                token = loaded_token

            embedding = embedding.to(dtype=text_encoder.dtype, device=text_encoder.device)

            # 3. Make sure we don't mess up the tokenizer or text encoder
            vocab = tokenizer.get_vocab()
            if token in vocab:
                raise ValueError(
                    f"Token {token} already in tokenizer vocabulary. Please choose a different token name or remove {token} and embedding from the tokenizer and text encoder."
                )
            elif f"{token}_1" in vocab:
                multi_vector_tokens = [token]
                i = 1
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    multi_vector_tokens.append(f"{token}_{i}")
                    i += 1

                raise ValueError(
                    f"Multi-vector Token {multi_vector_tokens} already in tokenizer vocabulary. Please choose a different token name or remove the {multi_vector_tokens} and embedding from the tokenizer and text encoder."
                )

            is_multi_vector = len(embedding.shape) > 1 and embedding.shape[0] > 1

            if is_multi_vector:
                tokens = [token] + [f"{token}_{i}" for i in range(1, embedding.shape[0])]
                embeddings = [e for e in embedding]  # noqa: C416
            else:
                tokens = [token]
                embeddings = [embedding[0]] if len(embedding.shape) > 1 else [embedding]

            # add tokens and get ids
            tokenizer.add_tokens(tokens)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            token_ids_and_embeddings += zip(token_ids, embeddings)

        # resize token embeddings and set all new embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))
        for token_id, embedding in token_ids_and_embeddings:
            text_encoder.get_input_embeddings().weight.data[token_id] = embedding
        
        return text_encoder, tokenizer



@threestudio.register("stable-diffusion-sds-bridge-prompt-processor")
class SDSBridgePromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pass

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device)

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder
        cleanup()

    def get_text_embeddings(
        self, prompt_src: Union[str, List[str]], prompt_tgt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B 77 768"], Float[Tensor, "B 77 768"]]:
        if isinstance(prompt_src, str):
            prompt_src = [prompt_src]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        # Tokenize text and get embeddings
        tokens_src = self.tokenizer(
            prompt_src,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        tokens_tgt = self.tokenizer(
            prompt_tgt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings_src = self.text_encoder(tokens_src.input_ids.to(self.device))[0]
            text_embeddings_tgt = self.text_encoder(tokens_tgt.input_ids.to(self.device))[0]
            uncond_text_embeddings = self.text_encoder(
                uncond_tokens.input_ids.to(self.device)
            )[0]

        return text_embeddings_src, text_embeddings_tgt, uncond_text_embeddings

    ###

    def spawn_func(self, pretrained_model_name_or_path, prompts, cache_dir):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            device_map="auto",
        )
        if len(self.cfg.texture_inversion_embedding) > 0:
            text_encoder, tokenizer = self.load_textual_inversion(self.cfg.texture_inversion_embedding, text_encoder, tokenizer)

        with torch.no_grad():
            tokens = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            text_embeddings = text_encoder(tokens.input_ids.to(text_encoder.device))[0]

        for prompt, embedding in zip(prompts, text_embeddings):
            torch.save(
                embedding,
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )

        del text_encoder
