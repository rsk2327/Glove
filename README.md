# Glove 
 Training of Glove embeddings for specific domains and to evaluate the benefit of using domain-specific data.
 
### Implementations : 
#### 1. Tensorflow
- Can be trained using Distributed learning 
- Trains Focal and Context vectors separately
#### 2. Cython
- Faster than the Tensorflow implementation. Parallelized using Cython’s built-in methods
- Trains only one version of vectors

***

### Results
For understanding the relationship between embeddings of words used in the financial context and other generic words, a curated list of words were selected for the evaluation task. The words in this list can broadly be classified into 3 groups :
- Purely financial terms Eg : Cash, Profit, Asset
- Financial terms with ambiguous meanings Eg : Market, deposit, bonds, interest
- Generic words : Strength, grocery, like, attention

#### Data Sources used : 
- Financial Data : Rueters Financial Dataset
- Generic Data : Wikipedia 2010 Dump

#### Pretrained Glove model
![Combined image examples](https://i.ibb.co/TH21rLK/Glove-pretrained-300.png)

A key observation was the characteristic band structure across the generic terms. This was expected as these words had no relationship ( low co-occurence) with the other, mostly financial terms. We also notice that the bottom right corner is more strongly highlighted as compared to the other regions. This is again expected since all purely financial terms are expected to have high co-occurence.

#### Pretrained Word2Vec model
![Combined image examples](https://i.ibb.co/Sm9b5MX/Word2-Vec-pretrained.png)


#### Glove model trained using Financial Data
![Combined image examples](https://i.ibb.co/YyyBnfS/Glove-findata-300.png)

Training with financial data gave 2 key observations. Firstly, the band structure which was very dominant in the pretrained model does not stand out in this model. Secondly, similarity of certain word-pairs used in the financial context become highlighted. Eg : The words mutual and funds are highlighted in this model whereas it wasnt in the pretrained models. 

#### Glove model trained using Financial Data + Generic Data 
<img src="https://i.ibb.co/Jk44qKY/Glove-combined-300.jpg" data-canonical-src="https://i.ibb.co/Jk44qKY/Glove-combined-300.jpg" width="550" height="550" />

By training a model on both Generic and Financial datasets, we could see that both generic as well as domain specific patterns (word-pair similarities) were retained in the combined model. This strongly suggests that training Glove vectors with domain specific datasets can generate more relavant word embeddings.

***

### Future Work
- Evaluate word vectors by using them as embeddings in domain-related NLP tasks. 
