from dataclasses import dataclass
import os


@dataclass
class ModelConfig:
    model_name: str = os.getenv("POLICYSIM_MODEL_NAME", "glm-4-flash")
    base_url: str = os.getenv("POLICYSIM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
    api_key: str = os.getenv("POLICYSIM_API_KEY", "")
    temperature: float = float(os.getenv("POLICYSIM_TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("POLICYSIM_MAX_TOKENS", "1024"))


@dataclass
class TrainConfig:
    base_model: str = os.getenv("POLICYSIM_TRAIN_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    output_dir: str = os.getenv("POLICYSIM_TRAIN_OUTPUT", "outputs")
    learning_rate: float = float(os.getenv("POLICYSIM_TRAIN_LR", "1e-5"))
    epochs: int = int(os.getenv("POLICYSIM_TRAIN_EPOCHS", "3"))
    batch_size: int = int(os.getenv("POLICYSIM_TRAIN_BATCH_SIZE", "2"))

