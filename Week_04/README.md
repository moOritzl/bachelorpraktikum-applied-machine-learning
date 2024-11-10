# Linear Classification with Gradient Descent and AdaGrad

## Overview

This project explores linear classification using two optimization techniques: Bold Driver and AdaGrad.

1. **Bold Driver**: A linear classifier using stochastic gradient descent with adaptive step length for efficient
   convergence.
2. **AdaGrad**: A linear classifier with adaptive step length per parameter, using different initial learning rates to
   assess performance.

## Data and Preprocessing

- **Datasets**: Bank Marketing and Occupancy Detection.
- **Steps**: Dropped irrelevant features, replaced missing values, applied One-Hot Encoding, and normalized data.

## Results

- **Bold Driver**: Smooth convergence, effective for both datasets with minimal tuning.
- **AdaGrad**: Sensitive to learning rates; showed mixed performance depending on initial rate.

## Conclusion

- **Bold Driver** is more stable and requires less tuning compared to **AdaGrad**, which is sensitive to learning rate
  choices.
