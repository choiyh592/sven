# Overview

### Exact Computation Overview
![](/img/shap_estimation_exact.png)

### Approximation Overview
![](/img/shap_approximation.png)

Development in process


# How to use
```python
from sven import ExactExplainer # or ApproxExplainer1

svenexplainer = ExactExplainer(torch_model, tensor_dataset, feature_vector_size, method='your_method', nan=0, device='cuda')

```