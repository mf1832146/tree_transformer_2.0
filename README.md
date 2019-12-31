tree-transformer
====
This project is build for testing tree-transformer on sentiment-classification and code summary.

code summary
----
### reading list
#### 1. A Survey of Automatic Generation of Source Code Comments: Algorithms and Techniques(https://arxiv.org/abs/1907.10863)
1. the categories of source code models
    * tokens-based models (key words or topics)
    * ast-based models

2. the categories of comment generation
    * rule-based generation solution
    * generative-based method (decoder)
    * search-based text generation (search comments existing in the corpus)

3. comment types
    * descriptive comments
    * summary comments
    * conditional comments
    * comments for debugging
    * metadata comments

4. algorithms
    * IR based algorithms: compute the relevance between target code and other source code in dataset, then generate comments based on the well matched codes' (one or multiple) comments. (LDA, VSM, LSI) very similar to the methods used in code clone detection
    * deep neural networks based comment generation algorithms

    * single-encoder-decoder : 
        >+ {CODE-NN}[https://www.aclweb.org/anthology/P16-1195.pdf] pro: using LSTM with attention as an encoder
        >- {GRU-NN}{https://arxiv.org/pdf/1709.07642.pdf} pro: using GRU with global attention as an encoder 
        >- {DeepCom}{https://xin-xia.github.io/publication/icpc182.pdf} : use sbt to generate the API call sequence.
        >- {Tree2Seq}{https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16492/16072} : use AST as input directly, with RvNN
 
    * multiple-encoder:
        >* {API+Code}[https://xin-xia.github.io/publication/ijcai18.pdf] using two encoders, use both code information and API information
        >* {RL+Hybrid2Seq}[https://arxiv.org/pdf/1811.07234.pdf]  AST-based LSTM + LSTM as encoders
 
    * CNN encoder:
        using CNN as an encoder {Suggesting accurate method and class names}

RL+Hybrid2Seq:
1. motivations: 
    * the structure information of code is ignored
    * traditional maximum likehood based methods suffer from the exposure bias issue.
2. contributions
    * use an AST-based LSTM to catch the structual information and an sequencial LSTM to deal with the code sequence. 
    * use an hybrid attention layer to fuse the two representations
    * use the deep reinforcement learning to deal with the exposure bias issue.
    
 
 
 


### 1. DataSet
### 2. Baseline Model
### 3. Valid method
