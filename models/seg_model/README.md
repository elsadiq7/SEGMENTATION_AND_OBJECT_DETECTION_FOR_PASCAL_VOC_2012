---
library_name: segmentation-models-pytorch
license: mit
pipeline_tag: image-segmentation
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
- segmentation-models-pytorch
- semantic-segmentation
- pytorch
languages:
- python
---
# Unet Model Card

Table of Contents:
- [Load trained model](#load-trained-model)
- [Model init parameters](#model-init-parameters)
- [Model metrics](#model-metrics)
- [Dataset](#dataset)

## Load trained model
```python
import segmentation_models_pytorch as smp

model = smp.from_pretrained("<save-directory-or-this-repo>")
```

## Model init parameters
```python
model_init_params = {
    "encoder_name": "resnet50",
    "encoder_depth": 5,
    "encoder_weights": "imagenet",
    "decoder_use_batchnorm": True,
    "decoder_channels": (256, 128, 64, 32, 16),
    "decoder_attention_type": None,
    "decoder_interpolation_mode": "nearest",
    "in_channels": 3,
    "classes": 21,
    "activation": "softmax",
    "aux_params": None
}
```

## Model metrics
```json
{
    "iou": 0.6300861239433289
}
```

## Dataset
Dataset name: pascal_voc

## More Information
- Library: https://github.com/qubvel/segmentation_models.pytorch
- Docs: https://smp.readthedocs.io/en/latest/

This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin)