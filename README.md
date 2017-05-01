# ML2016

My source code for the assignments of [professor Hung-Yi Lee's machine learning course in 2016/09 - 2017/01](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML16.html)

## Branches

* master: A branch containing the report and the best-performance scripts for each assignment.
* develop: A branch containing all the source code.

## Assignments

<table style="width:100%">
  <tr>
    <th></th>
    <th>Title</th>
    <th>Desc.</th>
    <th>Datasets</th>
    <th>Methods</th>
    <th>Results</th>
  </tr>
  <tr>
    <td>HW1</td>
    <td>Air Quality Prediction</td> 
    <td>
      <a href="https://docs.google.com/presentation/d/1gB0AGVGlPAZKJVnZiuqukb3Oc2vm0QHRq23Z2U0cJ_c/edit?usp=sharing">Slides</a>
      <br/>
      <a href="https://inclass.kaggle.com/c/ml2016-pm2-5-prediction">Kaggle</a>
    </td>
    <td>air quality data collected by Central Weather Bureau</td>
    <td>Linear Regression</td>
    <td>Top 11% (37th in 348 participants)</td>
  </tr>
  <tr>
    <td>HW2</td>
    <td>Spam Classification</td> 
    <td>
      <a href="https://docs.google.com/presentation/d/19ZAdqDiplehM0hNZw4vC5RLruS4Qhp_pzVVLrDAD_Pw/edit?usp=sharing">Slides</a>
      <br/>
      <a href="https://inclass.kaggle.com/c/spam-classification">Kaggle</a>
    </td>
    <td>4001(train) + 600(test) emails with extracted 57 features for each mail</td>
    <td>Logistic Regression, DNN</td>
    <td>Top 10% (27th in 276 participants)</td>
  </tr>
  <tr>
    <td>HW3</td>
    <td>Semi-supervised Learning in Picture Classification.</td> 
    <td>
      <a href="https://docs.google.com/presentation/d/1xYJG_QLSHrQcYwan6PBf_l0QYp3tJuDvpgrU37ovQeY/edit#slide=id.p">Slides</a>
      <br/>
      <a href="https://inclass.kaggle.com/c/ml2016-semi-supervised-learning">Kaggle</a>
    </td>
    <td>cifar-10 (5,000 labeled images + 45,000 unlabeled images + 10,000 test images)</td>
    <td>CNN, Semi-supervised learning, Autoencoder (Using Keras)</td>
    <td>Top 47% (121st in 258 participants)</td>
  </tr>
  <tr>
    <td>HW4</td>
    <td>Document Clustering</td> 
    <td>
      <a href="https://drive.google.com/file/d/0ByWxKDk6FoXoSnVPUFNBZXN2NUk/view">Slides</a>
      <br/>
      <a href="https://inclass.kaggle.com/c/ml2016-hw4-unsupervised-learning">Kaggle</a>
    </td>
    <td>20,000 StackOverflow documents crawled from the internet</td>
    <td>
      <ul>
        <li>Bag-of-Words</li>
        <li>TF-IDF</li>
        <li>Latent Semantic Analysis</li>
        <li>Word Vectors</li>
        <li>K-Means</li>
        <li>PCA</li>
        <li>t-SNE</li>
        Using scikit-learn, gensim
      </ul>
    </td>
    <td>Top 59% (156th in 263 participants)</td>
  </tr>
  <tr>
    <td>Final Project</td>
    <td>Transfer Learning on Stack Exchange Tags</td> 
    <td>
      <a href="https://docs.google.com/presentation/d/1Xe8oa6niPxPZwN0b_qmUGFPJb6K5Gd7wIrvqravvkNM/edit#slide=id.p">Slides</a>
      <br/>
      <a href="https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags">Kaggle</a>
    </td>
    <td>
      <ul>
        <li>Train: tagged Stack Exchange Questions in 6 domains</li>
        <li>Test: untagged Stack Exchange Questions in another domain</li>
      </ul>
    </td>
    <td>
      <ul>
        <li>Bag-of-Words</li>
        <li>TF-IDF</li>
        <li>Latent Semantic Analysis</li>
        <li>Word Vectors</li>
        <li>K-Means</li>
        <li>PCA</li>
        <li>t-SNE</li>
      </ul>
      (Using scikit-learn, Natural Language Toolkit, gensim)
    </td>
    <td>Top 30% (114th in 380 participants)</td>
  </tr>
</table>
