# Backpropagation Lab — Manual (NumPy) vs Automatic (TensorFlow)

Lab 2 of the Deep Learning course: implement backpropagation by hand with NumPy, then redo the
same model with TensorFlow's automatic differentiation and compare the two.

Dataset: [`sklearn.datasets.load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
(8×8 grayscale digit images, 10 classes).

## Contents

| Path | Description |
| --- | --- |
| [`notebooks/Lab2_Part1_Backpropagation_numpy.ipynb`](notebooks/Lab2_Part1_Backpropagation_numpy.ipynb) | **Part 1** — softmax, negative log-likelihood, logistic regression trained with SGD, then a one-hidden-layer sigmoid MLP with hand-derived gradients, verified by a finite-difference gradient check. |
| [`notebooks/Lab2_Part2_Backpropagation_tensorflow.ipynb`](notebooks/Lab2_Part2_Backpropagation_tensorflow.ipynb) | **Part 2** — TensorFlow basics, `tf.GradientTape` autodiff, and the same digit classifier written with the low-level TensorFlow API (no Keras), plus experiments on depth, initialization, ReLU and L2 regularization. |
| [`report/Lab2_Report.pdf`](report/Lab2_Report.pdf) | One-page report: manual vs TensorFlow backprop, where manual errors happen, what TensorFlow automates, and an example where autodiff wins. |

## Key results (test accuracy, 270 test images)

| Model / setting | Test acc. |
| --- | --- |
| NumPy logistic regression (1 epoch) | ≈ 95.6% |
| NumPy MLP, 10 sigmoid units, 15 epochs | 96.7% |
| NumPy MLP + momentum 0.9 / Keras (same setup) | 95.9% / 95.9% |
| NumPy MLP, zero init | 49.3% |
| TF MLP, 15 sigmoid units, 10 epochs | 95.9% |
| TF + L2 (λ = 1e-4), 50 units | 98.5% |
| TF ReLU [50, 20], He init | 98.2% |
| TF sigmoid [50, 20], init σ = 0.01 | 36.3% |

The NumPy gradients match `tf.GradientTape` to a relative difference of ~2·10⁻⁷.

## Running

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

The notebooks were written for Google Colab and also run locally with Python 3.10+.
