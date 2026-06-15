import os
import time


QWEN3_30B_A3B_INSTRUCT_2507 = "Qwen/Qwen3-30B-A3B-Instruct-2507"


class Generator:
    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        self.api_key = api_key

    def generate(self, prompt):
        return []


def _message_content(choice):
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message")

    if isinstance(message, dict):
        return message.get("content", "")

    return getattr(message, "content", "") or ""


def _apply_stop(text, stop):
    for stop_sequence in stop or []:
        if stop_sequence and stop_sequence in text:
            text = text.split(stop_sequence, 1)[0]
    return text


class OpenAIGenerator(Generator):
    def __init__(
        self,
        model_name,
        api_key,
        base_url=None,
        n=8,
        max_tokens=512,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
        wait_till_success=False,
    ):
        super().__init__(model_name, api_key)
        self.base_url = base_url
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop or ["\n\n\n"]
        self.wait_till_success = wait_till_success
        self.client = self._client_init()

    def _client_init(self):
        try:
            import openai
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "OpenAIGenerator requires the 'openai' package. "
                "Install project dependencies with `pip install -e .`."
            ) from e

        return openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    @staticmethod
    def parse_response(response):
        choices = response["choices"] if isinstance(response, dict) else response.choices
        return [_message_content(choice).strip() for choice in choices if _message_content(choice)]

    def _generate_once(self, prompt):
        return self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            top_p=self.top_p,
            stop=self.stop,
        )

    def generate(self, prompt):
        texts = []
        while len(texts) < self.n:
            try:
                result = self._generate_once(prompt)
                texts.extend(self.parse_response(result))
            except Exception as e:
                if self.wait_till_success:
                    time.sleep(1)
                else:
                    raise e
        return texts[: self.n]


class HuggingFaceGenerator(Generator):
    def __init__(
        self,
        model_name=QWEN3_30B_A3B_INSTRUCT_2507,
        api_key=None,
        provider="auto",
        base_url=None,
        n=8,
        max_tokens=512,
        temperature=0.7,
        top_p=0.8,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
        wait_till_success=False,
    ):
        api_key = api_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        super().__init__(model_name, api_key)
        self.provider = provider
        self.base_url = base_url
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop or ["\n\n\n"]
        self.wait_till_success = wait_till_success
        self.client = self._client_init()

    def _client_init(self):
        try:
            from huggingface_hub import InferenceClient
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "HuggingFaceGenerator requires the 'huggingface_hub' package. "
                "Install project dependencies with `pip install -e .`."
            ) from e

        return InferenceClient(
            model=self.model_name,
            provider=self.provider,
            base_url=self.base_url,
            api_key=self.api_key,
        )

    @staticmethod
    def parse_response(response):
        choices = response["choices"] if isinstance(response, dict) else response.choices
        return [_message_content(choice).strip() for choice in choices if _message_content(choice)]

    def _generate_once(self, prompt):
        return self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            top_p=self.top_p,
            stop=self.stop,
        )

    def generate(self, prompt):
        texts = []
        while len(texts) < self.n:
            try:
                result = self._generate_once(prompt)
                texts.extend(self.parse_response(result))
            except Exception as e:
                if self.wait_till_success:
                    time.sleep(1)
                else:
                    raise e
        return texts[: self.n]


def _is_hf_offline_mode():
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes", "on"}


class TransformersGenerator(Generator):
    def __init__(
        self,
        model_name=QWEN3_30B_A3B_INSTRUCT_2507,
        api_key=None,
        n=8,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8,
        stop=None,
        device_map="auto",
        torch_dtype="auto",
        cache_dir=None,
        trust_remote_code=False,
        attn_implementation=None,
        low_cpu_mem_usage=True,
        local_files_only=None,
    ):
        if api_key is None:
            api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        super().__init__(model_name, api_key)
        self.n = n
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop or ["\n\n\n"]
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir or os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME")
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.low_cpu_mem_usage = low_cpu_mem_usage
        if local_files_only is None:
            local_files_only = _is_hf_offline_mode()
        self.local_files_only = local_files_only
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.model is not None and self.tokenizer is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "TransformersGenerator requires the 'transformers' package. "
                "Install project dependencies with `pip install -e .`."
            ) from e

        common_kwargs = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if self.local_files_only:
            # Avoid using an expired token stored on disk when loading from cache.
            common_kwargs["token"] = False
        elif self.api_key:
            common_kwargs["token"] = self.api_key

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **common_kwargs)

        model_kwargs = {
            **common_kwargs,
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
        }
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

    def _input_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.model.device

    def _generate_once(self, prompt):
        self._load_model()
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt")
        input_length = model_inputs["input_ids"].shape[-1]
        input_device = self._input_device()
        model_inputs = {key: value.to(input_device) for key, value in model_inputs.items()}

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature and self.temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        try:
            import torch
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "TransformersGenerator requires PyTorch. Install a CUDA-enabled torch build "
                "for local Qwen generation."
            ) from e

        with torch.inference_mode():
            generated_ids = self.model.generate(**model_inputs, **generation_kwargs)

        output_ids = generated_ids[0][input_length:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return _apply_stop(text, self.stop).strip()

    def generate(self, prompt):
        texts = []
        while len(texts) < self.n:
            text = self._generate_once(prompt)
            if text:
                texts.append(text)
        return texts


class CohereGenerator(Generator):
    def __init__(
        self,
        model_name,
        api_key,
        n=8,
        max_tokens=512,
        temperature=0.7,
        p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
        wait_till_success=False,
    ):
        super().__init__(model_name, api_key)
        try:
            import cohere
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "CohereGenerator requires the 'cohere' package. "
                "Install project dependencies with `pip install -e .`."
            ) from e

        self.cohere = cohere.Cohere(self.api_key)
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.p = p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop or ["\n\n\n"]
        self.wait_till_success = wait_till_success

    @staticmethod
    def parse_response(response):
        text = response.generations[0].text
        return text

    def generate(self, prompt):
        texts = []
        for _ in range(self.n):
            get_result = False
            while not get_result:
                try:
                    result = self.cohere.generate(
                        prompt=prompt,
                        model=self.model_name,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty,
                        p=self.p,
                        k=0,
                        stop=self.stop,
                    )
                    get_result = True
                except Exception as e:
                    if self.wait_till_success:
                        time.sleep(1)
                    else:
                        raise e
            text = self.parse_response(result)
            texts.append(text)
        return texts
