# Overview

![](/img/shap_estimation.png)

Development in process


# How to use
```python
from sven.explainers import Explainer

svenexplainer = Explainer(torch_model, tensor_dataset, feature_vector_size, method='your_method', nan=0, device='cuda')

```