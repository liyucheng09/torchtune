from torchtune.models.deepseek_lite._model_builders import deepseek_small
import torch

model = deepseek_small()

out = model(
    torch.randint(0, 32_000, (2, 256)),
)

print(out.shape)