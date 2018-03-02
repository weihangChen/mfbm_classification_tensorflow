# mfbm_classification_tensorflow
## Communication Protocol Summary

The task is about building a communication protocol, given following user senario as example
* sender generate number sequence representing 64-characters.  (ex. "a" => "0.001, 0.0023...", "b" => "0.003, 0.4..." ) and send the sequences to receiver
* receiver decode the sequence back to 64-characters


Brownian motion can be used to generate correlated random numbers and the generated sequence is different at each realizationand. The correlation is derived from the covariance matrix and it makes classification possible, hence it is used in the sender side to generate sequence. What is more interesting is that we used to apply machine learning technique to gathered data, in this task, the data is manufactured. At the receiver side, we use CNN + Tensorflow to the sequence classifier. 


Two solutions are discussed in the second section, “correlated random number generation” and “machine learning training” are discussed in section three and four. Here are two simple videoes showing the encode, classify and decode process, first video is done in fbm and the second is done in mfbm.

<a href="https://youtu.be/AEjreQ62tzM" target="_blank"><img src="http://img.youtube.com/vi/AEjreQ62tzM/0.jpg" 
alt="fbm" width="240" height="180" border="10" /></a>

<a href="https://youtu.be/SntcxVjUj9A" target="_blank"><img src="http://img.youtube.com/vi/SntcxVjUj9A/1.jpg" 
alt="mfbm" width="240" height="180" border="10" /></a>

## Solution proposals
### Proposal1
Each alphabet is mapped to three digits and each digit is further mapped to one Hurst value ("0" corresponds to "H0.7", "1" corresponds to "H0.75" and "2" correponds to "H0.8"). For example, “a”=>”000”=>”H0.7/H0.7/H0.7”, “f”=>”012”=>”H0.7/H0.75/H0.8”. When alphabet “a” is sent, three Brownian motion random walk realizations using Hurst value 0.7 occur, and 256 * 3 digits in total is sent. The classifier classifies each 256 digits and get “000”, hence “a” is obtained. In this solution, Hurst value in each realization remains the same. With 256 as sequence length, we achieve a 95% accuracy on this solution with 3 labels. 

Hurst 0.8: 
![Figure - 0.8][fbm_h8]

[fbm_h8]: https://github.com/weihangChen/mfbm_classification_tensorflow/blob/master/mfbm/images/fbm_h8.JPG "fbm_h8"

Hurst 0.4:
![Figure - 0.4][fbm_h4]

[fbm_h4]: https://github.com/weihangChen/mfbm_classification_tensorflow/blob/master/mfbm/images/fbm_h4.JPG "fbm_h4"

### Proposal2
Here we utilize the characteristics of multi fractional Brownian motion where Hurst value is dynamic and each unique Hurst function corresponds to one label. 

multifractional realization for ramp function:
```python
def h1(t):
    if t <= 0.5:
        return 0.2
    if 0.5 < t <= 0.7:
        return 0.2 + (t - 0.5)
    return 0.4
```

![Figure - 0.9][mfbmh1]

[mfbmh1]: https://github.com/weihangChen/mfbm_classification_tensorflow/blob/master/mfbm/images/mfbm_h1.JPG "mfbm1"


mfbm realization for ramp function:
```python
def h2(t):
    if t <= 0.5:
        return 0.4
    if 0.5 < t <= 0.7:
        return 0.4 - (t - 0.5)
    return 0.25
```

![Figure - 1.9][mfbmh2]

[mfbmh2]: https://github.com/weihangChen/mfbm_classification_tensorflow/blob/master/mfbm/images/mfbm_h2.JPG "mfbm1"

## Generate correlated random digit sequence
This [wiki page](https://en.wikipedia.org/wiki/Fractional_Brownian_motion) and [youtube video](https://www.youtube.com/watch?v=QCqsJVS8p5A) provide a great overview of the whole process. This [github project](https://github.com/crflynn/fbm) contains the implementation for three different approaches – “daviesharte”, “cholesky” and “hosking”. 
([ref1](https://stats.stackexchange.com/questions/38856/how-to-generate-correlated-random-numbers-given-means-variances-and-degree-of))

* Step1: Generate random numbers from Gaussian distribution ([ref1](http://blog.csdn.net/lanchunhui/article/details/50163669), [ref2](https://www.youtube.com/watch?v=4PLJv84014I))

* Step2: create symmetric, positive-definite covariance matrix using Hurst value other than 0.5. ([ref1](http://stattrek.com/matrix-algebra/covariance-matrix.aspx), [ref2](http://comisef.wikidot.com/tutorial:correlation),[ref3](https://www.youtube.com/watch?v=0W8hTzU1ZMM),[ref4](https://www.youtube.com/watch?v=LmZAwtQ6XzI&t=238s))

* Step3: Generate the square root (standard deviation) matrix of covariance matrix from step2 using Cholesky decomposition and pick the lower left one. ([ref1](https://en.wikipedia.org/wiki/Fractional_Brownian_motion), [ref2](https://www.youtube.com/watch?v=gFaOa4M12KU),[ref3](https://www.youtube.com/watch?v=j1epLYdfqT4),[ref4 - geometry visualization](https://blogs.sas.com/content/iml/2012/02/08/use-the-cholesky-transformation-to-correlate-and-uncorrelate-variables.html))

* Step4: multiply random numbers (generated from step1) as single column vector with the standard deviation matrix (generated from step3). 

Obviously, there is math formular for that, but if you don't understand the part with covariance matrix, you probably won't figure out why classification is possible in this task, which is the tricky part. Otherwise everything is inside of this 
[wonderful project](https://github.com/crflynn/fbm)


## Training and Classification
Hurst value estimator ([paper1](https://arxiv.org/pdf/1201.4786.pdf), [paper2](https://www.diva-portal.org/smash/get/diva2:828116/FULLTEXT01.pdf)), ARIMA models or the implementation in this [project](https://github.com/PTRRupprecht/GenHurst) is a good fit for manual feature extraction, but I decided to go for unsupervised feature extraction + classification using CNN and Tensorflow. These resources ([ref1](https://burakhimmetoglu.com/2017/08/22/time-series-classification-with-tensorflow/), [ref2](https://mapr.com/blog/deep-learning-tensorflow/) and [ref3](https://blog.cardiogr.am/applying-artificial-intelligence-in-medicine-our-early-results-78bfe7605d32)) analyze time series data to solve classification problem or predict future value. 

**Solution1**: there are only three labels, realizations are made using Hurst0.7, Hurst0.75 and Hurst0.8. 10000 samples with sequence length 256 for each Hurst value are sampled as training and validation dataset; and the trained accuracy on validation dataset is 93% - 98% 

![Figure - 11][fbm_256_acc_26chars]

[fbm_256_acc_26chars]: https://github.com/weihangChen/mfbm_classification_tensorflow/blob/master/mfbm/images/fbm_256_acc_26chars.JPG "fbm_256_acc_26chars"



**Solution2**:
Sequence length is 600 and deep learning architecture remains the same, when Hurst value difference is lower than 0.05, achieved accuracy only lands on 70%. Once we let it goes up to 0.07 then the accuracy lands on 99% with 3 labels

![Figure - 12][mfmb_600_acc_3chars]

[mfmb_600_acc_3chars]: https://github.com/weihangChen/mfbm_classification_tensorflow/blob/master/mfbm/images/mfmb_600_acc_3chars.JPG "mfmb_600_acc_3chars"

## Source Code
There are two packages - "fbmsolution" and "mfbmsolution", each package contains two entry files "solution.cs" and "training.cs". They can be run seperately to debug through the code base. Note: [Tensorflow installation is required](https://www.tensorflow.org/install/) 

