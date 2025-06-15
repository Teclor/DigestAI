import requests
from config import config
from utils.logger_configurator import LoggerConfigurator
from ollama.prompt_template_factory import PromptTemplateFactory
from transformers import AutoTokenizer

class SummaryService:
    def __init__(
        self,
        ollama_host: str | None = None,
        model: str | None = None,
        tokenizer_name: str | None = None
    ):
        self.ollama_url = ollama_host or config.ollama.url
        self.model = model or config.ollama.model
        self.tokenizer_name = tokenizer_name or config.ollama.default_tokenizer or self.model
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                trust_remote_code=True
            )
            self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def limit_tokens(self, text: str, max_tokens: int) -> str:
        """
        Обрезает текст до заданного числа токенов.
        :param text: Входной текст
        :param max_tokens: Максимальное количество токенов
        :return: Обрезанный текст
        """
        if max_tokens <= 0:
            return ""

        tokens = self.tokenizer.tokenize(text)
        if len(tokens) <= max_tokens:
            return text

        limited_tokens = tokens[:max_tokens]
        limited_text = self.tokenizer.convert_tokens_to_string(limited_tokens)

        return limited_text

    def summarize(self, text: str, template_type: str = "message", need_template: bool = True) -> str:
        """
        :param text: Текст для пересказа
        :param template_type: Тип текста (используется для выбора шаблона)
        :return: Пересказанный текст от модели
        """

        logger_config = LoggerConfigurator("summary_service.json", log_format="json")
        logger = logger_config.get_logger("summarize")

        prompt = PromptTemplateFactory.get_prompt(template_type).format(text=text) if need_template else text
        logger.info(f"prompt {prompt}")
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            logger.info(f"Host {self.ollama_url} model {self.model}")
            response = requests.post(
                f"{self.ollama_url}",
                json=payload,
                timeout=config.ollama.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.RequestException as e:
            raise RuntimeError(f"Ошибка при обращении к Ollama: {e}")