# Machine Learning Papers and best sources to learn topics

## Aggregators
- https://42papers.com/
- https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap - roadmap for NN from basis to advance
- https://paperswithcode.com/ - papers with code
- https://andlukyane.com/blog/ - review of modern papers
- https://farid.one/kaggle-solutions/ - top kaggle solutions
- https://emacsway.github.io/ru/self-learning-for-software-engineer - #offtop computer science roadmap
- https://www.arxiv-vanity.com/ - convert arxiv papers to html with click references and ability to translate via browser
- http://jalammar.github.io - best visualisations and explanation for papers
- https://lena-voita.github.io/nlp_course.html - excelent NLP course pages
- http://nlpprogress.com/ - SOTA rank results for every NLP task

## Deep learning
### NLP
- [Word2Vec, 2013](https://arxiv.org/pdf/1301.3781.pdf)
- [Negative Sampling, Hierarchical Softmax in Word2Vec, 2013](https://arxiv.org/pdf/1310.4546.pdf)
- [FastText, 2016](https://arxiv.org/pdf/1607.04606.pdf)
- [Clear explanation FastText](https://amitness.com/2020/06/fasttext-embeddings/)
- [BERT, 2018](https://arxiv.org/abs/1810.04805)
- [SummaRuNNer](https://arxiv.org/pdf/1611.04230.pdf) - text summarization

### Computer vision
#### Segmentation
- [Image Segmentation Using Deep Learning: A Survey](https://arxiv.org/pdf/2001.05566.pdf)
#### Detection
- [Object Detection in 20 Years: A Survey](https://arxiv.org/abs/1905.05055)
- [SSD](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) - best tutorial
#### GAN
- [Generative Adversarial Nets (GAN)](https://arxiv.org/pdf/1406.2661.pdf)
- *[A Survey on Generative Adversarial Networks: Variants, Applications, and Training](https://arxiv.org/pdf/2006.05132.pdf)
#### Common
- [*Diffusion models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [*MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- [FaceNet, 2015](https://arxiv.org/pdf/1503.03832.pdf)
- *[Transformers for image recognition at scale](https://arxiv.org/pdf/2010.11929.pdf)
- *[DALL'E](https://arxiv.org/abs/2102.12092)
- *[Deep Learning vs. Traditional Computer Vision](https://arxiv.org/pdf/1910.13796.pdf)

### RNN
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - attention in machine translating with RNN

### RL
- [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/) - policy function algorithm
- *[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - Q-function algorithm

### Distilation knowledge
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- *[A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/pdf/2103.13630.pdf)
- *[Modeling Teacher-Student Techniques in Deep Neural Networks for Knowledge Distillation](https://arxiv.org/abs/1912.13179)

### Common
- [Batchnorm explanation](https://leimao.github.io/blog/Batch-Normalization/) - main idea for cv: norm over minibatch for each filter
- [Layernorm explanation](https://leimao.github.io/blog/Layer-Normalization/) - main idea for cv: norm over ONE sample in minibatch for each filter
- [Guide for learning rate schedulers](https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling#7.-CyclicLR---triangular2)
- [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) - tricks and guide for coding DL|ML
- [Deep Double Descent: Where Bigger Models and More Data Hurt](https://arxiv.org/abs/1912.02292)
- [Auto-Encoding Variational Bayes (VAE)](https://arxiv.org/pdf/1312.6114.pdf)
- [VAE notes](https://www.notion.so/08644d3c48c341f28c4a93703b419a6d#4fbfcb8ddd0f411a83d093ac8c28770d) - excelent explanation for VAE in russian
- *[Tutorial on Variational Autoencoders](https://arxiv.org/pdf/1606.05908.pdf)
- *[Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

### Transformers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - transformer
- [Explanation of "Attention Is All You Need"](http://jalammar.github.io/illustrated-transformer/) - best explanation of paper
- *[A Survey of Transformers](https://arxiv.org/abs/2106.04554)
- *[GPT-3](https://arxiv.org/pdf/2005.14165.pdf)


## Recommender systems
### MF
- [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) - SGD, MSE, implicit, clean dataset, explain recommendations
- [ALS distributed, notes](http://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf)
- [Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618.pdf) - LearnBPR, MF, bayes optimizing ranking function
- [WSABIE: Scaling Up To Large Vocabulary Image Annotation](http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf) - smart pairwise sampling, optimizing ranking function 
- *[Factorization Machine](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)

### Improving recommendations
- *[Improving recommendation lists through topic diversification](https://www.researchgate.net/publication/200110416_Improving_recommendation_lists_through_topic_diversification)
- *[Who likes it more Mining Worth-Recommending Items from Long Tails by Modeling Relative Preference](https://github.com/zzhaozeng/IRPapers/blob/master/Group5/Who%20likes%20it%20more%20Mining%20Worth-Recommending%20Items%20from%20Long%20Tails%20by%20Modeling%20Relative%20Preference..pdf)

### Using deep learning
- [Monolith: Real Time Recommendation System With
Collisionless Embedding Table](https://arxiv.org/abs/2209.07663) - tik-tok online learning
- *[Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf)
- [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/ru//pubs/archive/45530.pdf)
- [Real-time Personalization using Embeddings for Search Ranking at Airbnb (Airbnb 2018)](https://github.com/murufeng/awesome-papers/blob/master/Embedding/%5BAirbnb%20Embedding%5D%20Real-time%20Personalization%20using%20Embeddings%20for%20Search%20Ranking%20at%20Airbnb%20(Airbnb%202018).pdf)

## Statistics - A/B
- [Trustworthy Online Controlled Experiments](https://experimentguide.com) - A Practical Guide to A/B Testing
- [Practitioner’s Guide to Statistical Tests](https://medium.com/@vktech/practitioners-guide-to-statistical-tests-ed2d580ef04f) - tests for proportion, pitfalls
- [Possion bootstrap](https://www.unofficialgoogledatascience.com/2015/08/an-introduction-to-poisson-bootstrap26.html)
- *[Delta Method](https://arxiv.org/pdf/1803.06336.pdf)
- [Hypo testing bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Bootstrap_hypothesis_testing)
- [CUPED at A/B](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Linearization at A/B](https://www.researchgate.net/publication/322969314_Consistent_Transformation_of_Ratio_Metrics_for_Efficient_Online_Controlled_Experiments)
- [Overlapping Experiment Infrastructure: More, Better, Faster Experimentation](https://static.googleusercontent.com/media/research.google.com/ru//pubs/archive/36500.pdf) - several experiments for one user at once

## Web search
- [Identifying Ambiguous Queries in Web Search](https://www.www2007.org/posters/poster941.pdf)
- [Determining the user intent of web search engine queries](https://www.researchgate.net/publication/221023370_Determining_the_user_intent_of_web_search_engine_queries)

## Uplift
- *[Causal Inference and Uplift Modeling. A review of the literature](http://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf)
- [Adapting Neural Networks for Uplift Models](marked_articles/2011.00041.pdf)
- *[A Practically Competitive and Provably Consistent Algorithm for Uplift Modeling](https://arxiv.org/pdf/1709.03683.pdf)

## Pricing
- *[Pricing Promotional Products Under Upselling](https://ziya.web.unc.edu/wp-content/uploads/sites/15166/2018/02/Aydin-Ziya-2008.pdf)

## Advertising
- [Practical Lessons from Predicting Clicks on Ads at Facebook](https://research.fb.com/wp-content/uploads/2016/11/practical-lessons-from-predicting-clicks-on-ads-at-facebook.pdf) - _downsampling_, features from GBM for LogReg, data freshness
- [Smart Pacing for Effective Online Ad Campaign Optimization](https://arxiv.org/pdf/1506.05851.pdf)

## GradientBoosting
- [XGBoost](https://arxiv.org/pdf/1603.02754.pdf)
- [CatBoost](https://arxiv.org/pdf/1706.09516.pdf)
- [CatBoost|XGBoost|LightGBM comparison](https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db)

## Metrics
- [F1-score, ROC_AUC, PR-AUC, comparison, prons and cons](https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc#1)
- [NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
- [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) - Mean_reciprocal_rank

## TimeSeries
- [Kats](https://github.com/facebookresearch/Kats) - time series library (better than prophet)
