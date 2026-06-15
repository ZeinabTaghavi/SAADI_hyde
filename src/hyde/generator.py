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
