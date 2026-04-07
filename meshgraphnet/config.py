from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    num_layers: int = 3
    hidden_dim: int = 32


@dataclass
class TrainingConfig:
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    num_epochs: int = 5


@dataclass
class Config:
    device: str = "cpu"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)