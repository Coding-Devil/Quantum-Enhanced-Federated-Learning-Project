from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelConfig:
    n_qubits: int = 4
    feature_dim: int = 128
    bert_model: str = 'bert-base-uncased'
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    learning_rate: float = 0.001
    n_epochs: int = 10
    privacy_epsilon: float = 0.1