# mfbm_classification_tensorflow
## Communication Protocol Summary
We used to apply deep learning to data that are given, in this task, we will try to generate data and covariance matrix ourselves then apply deep learning on the generated data. Text input is translated into digit sequences using multifractional Brownian motion and gets sent. Each such digit sequence is a realization of random walk and it is different at each realization. One important characteristics is that the generated digits in each sequence are correlated random numbers, and it makes classification possible. Received digit sequences are classified and get translated back to text content using the model built in CNN and Tensorflow. At technical level, we need to solve three problems. 

* Generate “correlated random number”.
* We need to modify this [fmb project](https://github.com/crflynn/fbm) and make it multifractional. 
* Apply machine learning technique to the generated fbm realizations.

Two solutions are discussed in the second section, “correlated random number generation” and “machine learning training” are discussed in section three and four.

## Solution proposals
### Proposal1
Each alphabet is mapped to three digits and each digit is further mapped to one Hurst value. For example, “a”=>”000”=>”H0.7/H0.7/H0.7”, “f”=>”012”=>”H0.7/H0.75/H0.8”. When alphabet “a” is sent, three Brownian motion random walk realizations using Hurst value 0.7 occur, and 256 * 3 digits in total is sent. The classifier classifies each 256 digits and get “000”, hence “a” is obtained. In this solution, Hurst value in each realization remains the same. With 256 as sequence length, we achieve a 95% accuracy on this solution. 


![Figure - 0.8][fbm_h8]

[fbm_h8]: https://github.com/weihangChen/mfbm_classification_tensorflow/blob/master/mfbm/images/fbm_h8.JPG "fbm_h8"
