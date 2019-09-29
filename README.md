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


#### Pretrained Word2Vec model
![Combined image examples](https://i.ibb.co/Sm9b5MX/Word2-Vec-pretrained.png)


#### Glove model trained using Financial Data
![Combined image examples](https://i.ibb.co/YyyBnfS/Glove-findata-300.png)

#### Glove model trained using Financial Data + Generic Data 
<img src="https://i.ibb.co/Jk44qKY/Glove-combined-300.jpg" data-canonical-src="https://i.ibb.co/Jk44qKY/Glove-combined-300.jpg" width="550" height="550" />
