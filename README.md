# ML-Models-Practice
Practice on making ML models with different frameworks
## Multiple Linear Regression

Predicts student's grade based on hours studied per week, previous scores, if they do extracurricular activities, average sleep hours per day, sample question papers practiced.

### PyTorch Large MSE Solution

Used gradient descent - iterative approximation that may not converge perfectly.

Gradient descent is sensitive to feature scales, causes the optimizer to struggle with convergence.

Solution : Replaced the gradient descent approach with the exact same mathematical formula that scikit-learn uses:

### Final Results Summary

| Framework    | Validation MSE | Test MSE |
| -------- | -------        |------- |
| Scikit-learn  | 4.07      | 4.07   |
| PyTorch    | 4.07         | 4.07 |
| TensorFlow | 4.12         | 4.17 |

## Logistic Regression

Predict if a patient needs dialysis based on their age, creatinine level, BUN, if they have diabetes, if they have hypertension, GFR,urine output, if patient has chronic kidney disease.

### Final Result Summary

| Framework  | Validation Accuracy | Test Accuracy      |
|------------|---------------------|--------------------|
| Sk-Learn   | 1.0                 | 0.9971098265895953 |
| Pytorch    | 0.9797688126564026  | 0.9942196607589722 |
| TensorFlow | 0.9942196607589722  | 0.9942196607589722 |




