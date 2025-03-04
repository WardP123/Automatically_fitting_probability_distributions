# Automatically_fitting_probability_distributions
This repository contains the code used for my Bachelor's thesis, titled "Automatically fitting probability distributions to data samples using a Convolutional Neural Network".
The main research question of the thesis is "To what extent is a neural network able to make accurate predictions about the probability distribution of sample-data?". The code was used to create a CNN with the Python TensorFlow library. The CNN was trained on artificially generated data sampled from probability distributions using the Python Nummpy Library. The code is divided into three different experiments. 

The 'CNN summary.png' contains a summary of the CNN architecture as returned by the TensorFlow summary function.

## Research Abstract
Researchers use probability distributions in a wide range of fields to identify patterns and probabilities in a data set. The process of fitting a probability distribution to data is called probability distribution fitting. The most common tool to fit a probability distribution to data is the goodness-of-fit test. However, the performance and reliability of goodness-of-fit tests highly depend on the type of data. This paper proposes a new, automated probability distribution fitting method by using a Convolutional Neural Network. The performance of the Convolutional Neural Network in finding the best-fit probability distribution is first tested on varying feature set lengths. Then, the network's power is analysed on distinguishing near-identical distributions. The results show that taking 400 samples of a distribution produces the optimal balance between the length of the feature set and the performance. Additionally, using a feature set length of 400, the network shows a strong performance on distinguishing similar distributions. Finally, the performance of the Convolutional Neural Network in classifying real-world data sets is examined. The results indicate that the network struggles to identify the probability distribution of real-world data.

# Probability distributions
For the hyperparameter optimisaton and the first experiment, the following probability distributions are used. The second- and third experiment use a subset of these distributions. 
* Beta
* Binomial
* Chi-squared
* Exponential
* Gamma
* Geometric
* Hypergeometric
* Lognormal
* Normal
* Poisson
* Student's t
* Uniform
* Weibull

## Experiment 1
The first experiment consists of finding the optimal length of the input vector, i.e. the number of samples per probability distribution. If the neural network does not receive enough samples of a probability distribution, it will not get enough data to accurately train the model. However, an input vector that is unnecessarily long will only burden the training process. The performance of the different input-vector lengths will be evaluated and compared. The length of the input vectors will be ranged from 50 to 1600 features. 

## Experiment 2
To further investigate the power of the model, the second experiment focuses on a subset of the distributions used above. For this experiment, only the Exponential, Gamma and Weibull distribution are used. The Weibull and Gamma distributions are generalizations of the Exponential distribution. Looking at their respective Probability Density Function (PDF) formulas shows that a specific combination of parameters makes all three distributions equal. The parameters are selected so that all distributions have equal PDFs. Then, one of the parameters will be step-wise increased by 0.1, which causes the distributions to gradually become distinguishable. The focus of the experiment is to identify from which point the classifier can correctly distinguish between these similar probability distributions.

## Experiment 3
In the final experiment, the classifier will be tested on real-world data. The real-world data is taken from Kaggle. The model will be tested on four data sets of which the data is known to follow a specific distribution. For example, people's heights is known to follow a normal distribution. The used data sets can be found in the Data directory. Once again, the training data set will be generated with artificial samples from probability distributions. However, for this experiment, the training data will be generated with parameters derived from the real-world data. Thus, the model will be trained four distinct times, with four different sets of parameters. The parameters will be calculated on 10\% of the real-world data set using the Maximum Likelihood Estimation (MLE) provided by the Scipy library
