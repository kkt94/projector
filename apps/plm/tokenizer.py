# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, cast

import tiktoken
import torch
from tiktoken.load import load_tiktoken_bpe

from core.data.conversation import REGISTERED_CONVS
from core.tokenizer import TikTokenTokenizer, Tokenizer

# Import HuggingFace transformers for Qwen tokenizer
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


class Llama3Tokenizer(TikTokenTokenizer):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|image|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # End of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        logger.info(f"Reloaded tiktoken model from {model_path}")

        self.n_words: int = self.model.n_vocab
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.eot_id: int = self.special_tokens["<|eot_id|>"]
        self.pad_id: int = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(
        self,
        s: str,
        add_bos: bool,
        add_eos: bool,
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            add_bos (bool): Whether to prepend the beginning-of-sequence token.
            add_eos (bool): Whether to append the end-of-sequence token.

        Returns:
            list[int]: A list of token IDs.
        """
        assert type(s) is str

        # The tiktoken tokenizer can handle <=400k chars without
        # pyo3_runtime.PanicException.
        TIKTOKEN_MAX_ENCODE_CHARS = 4000_000

        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consecutive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 250_000

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special="all",
                    disallowed_special=(),
                )
            )
        if add_bos:
            t.insert(0, self.bos_id)
        if add_eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]


@dataclass
class PLMTokenizedSample:
    is_valid: bool = True
    text_ids: List[int] = field(default_factory=list)
    image_pos: List[int] = field(default_factory=list)
    response_pos: List[int] = field(default_factory=list)
    num_media_chunks: int = 0


# Note that PLM using LLaMA (3.1 and 3.2) as base LLM.
class PLMTokenizer(Llama3Tokenizer):
    def __init__(
        self,
        model_path: str,
        patch_size: Optional[int] = None,
        pooling_ratio: Optional[float] = None,
        seq_len: Optional[int] = 2048,
        conversation_format: Optional[str] = "plm_sft",
        image_token: Optional[str] = "<|image|>",
        bos_token: Optional[str] = "<|begin_of_text|>",
        eos_token: Optional[str] = "<|end_of_text|>",
        num_visual_tokens: Optional[int] = None,
    ):
        super().__init__(model_path=model_path)

        self.patch_size = patch_size
        self.pooling_ratio = pooling_ratio
        self.seq_len = seq_len
        self.num_visual_tokens = num_visual_tokens
        self.conversation_template = REGISTERED_CONVS[conversation_format]
        self.image_token = image_token

        self.bos_token_id = self.special_tokens[bos_token]
        self.eos_token_id = self.special_tokens[eos_token]
        self.pad_token_id = self.pad_id
        self.image_token_id = self.special_tokens[self.image_token]
        self.eos_id = self.eos_token_id
        self.n_words = self.n_words

    def __call__(
        self,
        conversations: List[Any],
        media: Optional[torch.Tensor] = None,
        media_type: Optional[str] = "image",
    ) -> PLMTokenizedSample:
        conv_template = self.conversation_template.copy()
        assert self.image_token == conv_template.image_token
        conv_template.add_conv(deepcopy(conversations))

        if media_type in ["image", "multi_image", "video"]:
            if self.patch_size is not None and self.pooling_ratio is not None:
                # Standard patch-based token calculation
                num_media_chunks = media.size(0)
                width, height = media.size(-2), media.size(-1)
                num_patches = int(
                    (width // self.patch_size // self.pooling_ratio)
                    * (height // self.patch_size // self.pooling_ratio)
                )
                num_images_for_tokens = num_media_chunks
            else:
                # Fixed token count mode (CRT=294, A2=64, etc.)
                # Treat entire video as 1 chunk
                num_media_chunks = 1
                num_patches = self.num_visual_tokens if self.num_visual_tokens else 294
                num_images_for_tokens = 1
            dialog = conv_template.get_conversation_dict_list(
                num_images=num_images_for_tokens,
                num_patches=num_patches,
                media_type=media_type,
            )
        elif media_type == "text":
            # This is text-only sample
            num_media_chunks = 0
            dialog = conv_template.get_conversation_dict_list(
                num_images=0,
                num_patches=0,
                media_type=media_type,
            )
        else:
            NotImplementedError(
                f"The supported media types are ['image', 'multi_image', 'video', 'text'], \
                                but found {media_type} which is not supported"
            )

        text_ids = []
        source_ids = []
        response_ids = []
        response_pos = []
        for msg in dialog:
            for role, text in msg.items():
                tokens = self.encode(text, add_bos=False, add_eos=False)
                if role == "assistant":
                    response_ids.extend(tokens)
                else:
                    source_ids.extend(tokens)
                if (
                    len(text_ids) + len(source_ids) + len(response_ids) + 1
                    > self.seq_len
                ):
                    if len(text_ids) == 0:
                        return PLMTokenizedSample(is_valid=False)
                    logger.info(f"Truncated text length to {len(text_ids) + 1}")
                    break
                text_ids.extend(source_ids)
                response_pos.extend(
                    [i + len(text_ids) for i in range(len(response_ids))]
                )
                text_ids.extend(response_ids)
                source_ids = []
                response_ids = []

        image_pos = [i for i, t in enumerate(text_ids) if t == self.image_token_id]
        return PLMTokenizedSample(
            text_ids=text_ids,
            image_pos=image_pos,
            response_pos=response_pos,
            num_media_chunks=num_media_chunks,
        )

    def _tokenize_for_generation(
        self,
        question: List[Any],
        media: Optional[torch.Tensor] = None,
    ):
        if media is not None:
            width, height = media.size(-2), media.size(-1)
            num_patches = int(
                (width // self.patch_size // self.pooling_ratio)
                * (height // self.patch_size // self.pooling_ratio)
            )
            prompt = self.conversation_template.get_generation_prompt(
                question, num_images=len(media), num_patches=num_patches
            )
            text_ids = self.encode(prompt, add_bos=False, add_eos=False)
            image_pos = [i for i, t in enumerate(text_ids) if t == self.image_token_id]
        else:
            raise NotImplementedError(f"Text-only inference is not supported yet.")

        return text_ids, image_pos

    def decode_batch(self, tokens: torch.Tensor) -> List[str]:
        return [self.decode(tokens[i].tolist()) for i in range(tokens.size(0))]


class QwenTokenizer(Tokenizer):
    """
    Tokenizer for Qwen models using HuggingFace transformers.
    Supports Qwen2, Qwen2.5, and Qwen3 models.
    """

    def __init__(self, model_path: str):
        """
        Initializes the Qwen Tokenizer.

        Args:
            model_path (str): The HuggingFace model path (e.g., "Qwen/Qwen3-1.7B").
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers package is required for QwenTokenizer. Install with: pip install transformers")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.n_words: int = self.tokenizer.vocab_size
        self.bos_id: int = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 0
        self.eos_id: int = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        self.pad_id: int = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_id
        self.eot_id: int = self.eos_id  # End of turn token

        # Add image token if not present
        if "<|image|>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|image|>"]})
            self.n_words = len(self.tokenizer)

        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image|>")

        self.stop_tokens = {self.eos_id}

        logger.info(f"Loaded Qwen tokenizer from {model_path}")
        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

    def encode(
        self,
        s: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.
        """
        tokens = self.tokenizer.encode(s, add_special_tokens=False)
        if add_bos and self.bos_id is not None:
            tokens = [self.bos_id] + tokens
        if add_eos and self.eos_id is not None:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.
        """
        return self.tokenizer.decode(t, skip_special_tokens=False)

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        """Return the offsets of the tokens in the original text."""
        if tokens is None:
            tokens = self.encode(text, add_bos=False, add_eos=False)

        # Use HuggingFace tokenizer's offset mapping if available
        encoding = self.tokenizer(text, return_offsets_mapping=True)
        if 'offset_mapping' in encoding:
            offsets = [o[0] for o in encoding['offset_mapping']]
            substrs = [text[o[0]:o[1]] for o in encoding['offset_mapping']]
            return substrs, offsets

        # Fallback: simple character-based offset estimation
        substrs = []
        offsets = []
        current_pos = 0
        for token_id in tokens:
            token_str = self.tokenizer.decode([token_id])
            substrs.append(token_str)
            offsets.append(current_pos)
            current_pos += len(token_str)
        return substrs, offsets


class PLMQwenTokenizer(QwenTokenizer):
    """
    PLM-style tokenizer for Qwen models with conversation support.
    """

    def __init__(
        self,
        model_path: str,
        patch_size: Optional[int] = None,
        pooling_ratio: Optional[float] = None,
        seq_len: Optional[int] = 2048,
        conversation_format: Optional[str] = "plm_sft",
        image_token: Optional[str] = "<|image|>",
        num_visual_tokens: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model_path=model_path)

        self.patch_size = patch_size
        self.pooling_ratio = pooling_ratio
        self.seq_len = seq_len
        self.num_visual_tokens = num_visual_tokens
        self.conversation_template = REGISTERED_CONVS[conversation_format]
        self.image_token = image_token

        self.bos_token_id = self.bos_id
        self.eos_token_id = self.eos_id
        self.pad_token_id = self.pad_id

    def __call__(
        self,
        conversations: List[Any],
        media: Optional[torch.Tensor] = None,
        media_type: Optional[str] = "image",
    ) -> PLMTokenizedSample:
        conv_template = self.conversation_template.copy()
        # Handle image token mismatch
        if hasattr(conv_template, 'image_token'):
            conv_template.image_token = self.image_token
        conv_template.add_conv(deepcopy(conversations))

        if media_type in ["image", "multi_image", "video"] and media is not None:
            if self.patch_size is not None and self.pooling_ratio is not None:
                num_media_chunks = media.size(0)
                width, height = media.size(-2), media.size(-1)
                num_patches = int(
                    (width // self.patch_size // self.pooling_ratio)
                    * (height // self.patch_size // self.pooling_ratio)
                )
                num_images_for_dialog = num_media_chunks
            else:
                # Fixed token count mode (CRT=294, A2=64, etc.)
                num_media_chunks = 1
                num_patches = self.num_visual_tokens if self.num_visual_tokens else 294
                num_images_for_dialog = 1
            dialog = conv_template.get_conversation_dict_list(
                num_images=num_images_for_dialog,
                num_patches=num_patches,
                media_type=media_type,
            )
        elif media_type == "text":
            num_media_chunks = 0
            dialog = conv_template.get_conversation_dict_list(
                num_images=0,
                num_patches=0,
                media_type=media_type,
            )
        else:
            raise NotImplementedError(
                f"The supported media types are ['image', 'multi_image', 'video', 'text'], "
                f"but found {media_type} which is not supported"
            )

        text_ids = []
        source_ids = []
        response_ids = []
        response_pos = []
        for msg in dialog:
            for role, text in msg.items():
                tokens = self.encode(text, add_bos=False, add_eos=False)
                if role == "assistant":
                    response_ids.extend(tokens)
                else:
                    source_ids.extend(tokens)
                if (
                    len(text_ids) + len(source_ids) + len(response_ids) + 1
                    > self.seq_len
                ):
                    if len(text_ids) == 0:
                        return PLMTokenizedSample(is_valid=False)
                    logger.info(f"Truncated text length to {len(text_ids) + 1}")
                    break
                text_ids.extend(source_ids)
                response_pos.extend(
                    [i + len(text_ids) for i in range(len(response_ids))]
                )
                text_ids.extend(response_ids)
                source_ids = []
                response_ids = []

        image_pos = [i for i, t in enumerate(text_ids) if t == self.image_token_id]
        return PLMTokenizedSample(
            text_ids=text_ids,
            image_pos=image_pos,
            response_pos=response_pos,
            num_media_chunks=num_media_chunks,
        )

    def decode_batch(self, tokens: torch.Tensor) -> List[str]:
        return [self.decode(tokens[i].tolist()) for i in range(tokens.size(0))]


def build_tokenizer(name: str, path: str, **kwargs) -> Tokenizer:
    if name == "llama3":
        return Llama3Tokenizer(path)
    elif name == "plmchat":
        return PLMTokenizer(path, **kwargs)
    elif name == "qwen":
        return PLMQwenTokenizer(path, **kwargs)
    else:
        raise NotImplementedError(f"{name} tokenizer type is not implemented")
