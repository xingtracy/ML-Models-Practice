# ML-Models-Practice
Practice on making ML models with different frameworks
## Multiple Linear Regression

### PyTorch Large MSE Solution

Used gradient descent - iterative approximation that may not converge perfectly.

Gradient descent is sensitive to feature scales, causes the optimizer to struggle with convergence.

Solution : Replaced the gradient descent approach with the exact same mathematical formula that scikit-learn uses:

## Final Results Summary

| Model    | Validation MSE | Test MSE |
| -------- | -------        |------- |
| Scikit-learn  | 4.07      | 4.07   |
| TensorFlow | 4.12         | 4.17 |
| PyTorch    | 4.07         | 4.07 |
