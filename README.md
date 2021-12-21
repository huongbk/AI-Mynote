# AI Note

---

### Deep Learning Basic

- Sumary content from Deep learning basic course [AI4E](https://www.facebook.com/nttuan8.AI4E/)

---

- [x] Introduce ML

- [x] Linear regression

- [ ] Logistic regression

- [ ] Neural network

- [ ] Convolutional neural network

- [ ] CNN Techniques

- [ ] Autoencoder

- [ ] GAN

- [ ] Object detection

- [ ] RNN

---

#### Introduce ML

---

- [What is Artificial Intelligent?](https://www.ibm.com/cloud/learn/what-is-artificial-intelligence#:~:text=the%20human%20mind.-,What%20is%20artificial%20intelligence%3F,-While%20a%20number)

  - It is the science and engineering of making intelligent machines, especially intelligent computer programs. It is related to the similar task of using computers to understand human intelligence, but AI does not have to confine itself to methods that are biologically observable.
  - History of Artificial Intelligent: [_Key dates and names_](https://www.ibm.com/cloud/learn/what-is-artificial-intelligence#toc-history-of--2jPgsXm)

- [What is Machine learning?](https://www.ibm.com/cloud/learn/machine-learning#:~:text=within%20businesses%20today.-,What%20is%20machine%20learning%3F,-Machine%20learning%20is)

  - Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.

- [How machine learning works?](https://www.ibm.com/cloud/learn/machine-learning#toc-how-machin-NoVMSZI_)

  - UC Berkeley (link resides outside IBM) breaks out the learning system of a machine learning algorithm into three main parts.
    - A Decision Process: In general, machine learning algorithms are used to make a prediction or classification. Based on some input data, which can be labelled or unlabeled, your algorithm will produce an estimate about a pattern in the data.
    - An Error Function: An error function serves to evaluate the prediction of the model. If there are known examples, an error function can make a comparison to assess the accuracy of the model.
    - An Model Optimization Process: If the model can fit better to the data points in the training set, then weights are adjusted to reduce the discrepancy between the known example and the model estimate. The algorithm will repeat this evaluate and optimize process, updating weights autonomously until a threshold of accuracy has been met.

- Machine learning methods from [IBM Blog]()

  - [Supervised learning](https://www.ibm.com/cloud/learn/supervised-learning)

    - Supervised learning, also known as supervised machine learning, is a subcategory of machine learning and artificial intelligence. It is defined by its use of labeled datasets to train algorithms that to classify data or predict outcomes accurately.
    - As input data is fed into the model, it adjusts its weights until the model has been fitted appropriately, which occurs as part of the cross validation process.
    - Supervised learning helps organizations solve for a variety of real-world problems at scale, such as classifying spam in a separate folder from your inbox.

    - [_How supervised learning works?_](https://www.ibm.com/cloud/learn/supervised-learning#toc-how-superv-A-QjXQz-)

      - Supervised learning uses a training set to teach models to yield the desired output. This training dataset includes inputs and correct outputs, which allow the model to learn over time. The algorithm measures its accuracy through the loss function, adjusting until the error has been sufficiently minimized.
      - Supervised learning can be separated into two types of problems when data mining are classification and regression:
        - _Classification uses an algorithm to accurately assign test data into specific categories_. It recognizes specific entities within the dataset and attempts to draw some conclusions on how those entities should be labeled or defined. Common classification algorithms are linear classifiers, support vector machines (SVM), decision trees, k-nearest neighbor, and random forest, which are described in more detail below.
        - _Regression is used to understand the relationship between dependent and independent variables_. It is commonly used to make projections, such as for sales revenue for a given business. Linear regression, logistical regression, and polynomial regression are popular regression algorithms.

    - Supervised learning examples

      - Image- and object-recognition
      - Predictive analytics
      - Customer sentiment analysis
      - Spam detector

    - [_Supervised learning algorithms_](https://www.ibm.com/cloud/learn/supervised-learning#toc-supervised-QVA1W1YW)

      - _Neural network_

        - Primarily leveraged for deep learning algorithms, neural networks process training data by mimicking the interconnectivity of the human brain through layers of nodes.
        - Each node is made up of inputs, weights, a bias (or threshold), and an output. If that output value exceeds a given threshold, it “fires” or activates the node, passing data to the next layer in the network. Neural networks learn this mapping function through supervised learning, adjusting based on the loss function through the process of gradient descent.
        - When the cost function is at or near zero, we can be confident in the model’s accuracy to yield the correct answer.

      - _Navie bayes_

        - Naive Bayes is classification approach that adopts the principle of class conditional independence from the Bayes Theorem.
        - This means that the presence of one feature does not impact the presence of another in the probability of a given outcome, and each predictor has an equal effect on that result.
        - There are three types of Naïve Bayes classifiers: Multinomial Naïve Bayes, Bernoulli Naïve Bayes, and Gaussian Naïve Bayes. This technique is primarily used in text classification, spam identification, and recommendation systems.

      - _Linear regression_

        - Linear regression is used to identify the relationship between a dependent variable and one or more independent variables and is typically leveraged to make predictions about future outcomes.
        - When there is only one [independent variable](https://www.thoughtco.com/independent-and-dependent-variable-examples-606828#:~:text=to%20graph%20them.-,Independent%20Variable,-The%20independent%20variable) and one [dependent variable](https://www.thoughtco.com/independent-and-dependent-variable-examples-606828#:~:text=of%20the%20experiment.-,Dependent%20Variable,-The%20dependent%20variable), it is known as simple linear regression.
        - As the number of independent variables increases, it is referred to as multiple linear regression. For each type of linear regression, it seeks to plot a line of best fit, which is calculated through the method of least squares. However, unlike other regression models, this line is straight when plotted on a graph.

      - _Logistic regression_

        - While linear regression is leveraged when dependent variables are continuous, logistical regression is selected when the dependent variable is categorical, meaning they have binary outputs, such as "true" and "false" or "yes" and "no".
        - While both regression models seek to understand relationships between data inputs, logistic regression is mainly used to solve binary classification problems, such as spam identification.

      - _Suport vecter machine (SVM)_

        - A support vector machine is a popular supervised learning model developed by Vladimir Vapnik, used for both data classification and regression.
        - That said, it is typically leveraged for classification problems, constructing a hyperplane where the distance between two classes of data points is at its maximum. This hyperplane is known as the decision boundary, separating the classes of data points (e.g., oranges vs. apples) on either side of the plane.
        - Reference Support Vector Regression Tutorial for Machine Learning on [Analytic Vidhya](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/)

      - [_K-nearest neighbor_](https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning)

        - K-nearest neighbor, also known as the KNN algorithm, is a non-parametric algorithm that classifies data points based on their proximity and association to other available data. This algorithm assumes that similar data points can be found near each other. As a result, it seeks to calculate the distance between data points, usually through Euclidean distance, and then it assigns a category based on the most frequent category or average.
        - Its ease of use and low calculation time make it a preferred algorithm by data scientists, but as the test dataset grows, the processing time lengthens, making it less appealing for classification tasks. KNN is typically used for recommendation engines and image recognition.

      - [_Random forest_](https://www.ibm.com/cloud/learn/random-forest?mhsrc=ibmsearch_a&mhq=random%20forest#:~:text=your%20business%20goals.-,What%20is%20random%20forest%3F,-Random%20forest%20is)

        - Random forest is a commonly-used machine learning algorithm trademarked by Leo Breiman and Adele Cutler, which combines the output of multiple decision trees to reach a single result. Its ease of use and flexibility have fueled its adoption, as it handles both classification and regression problems.

  - [Unsupervised learning](https://www.ibm.com/cloud/learn/unsupervised-learning)

    - Unsupervised learning, also known as unsupervised machine learning, uses machine learning algorithms to analyze and cluster unlabeled datasets. These algorithms discover hidden patterns or data groupings without the need for human intervention.
    - Its ability to discover similarities and differences in information make it the ideal solution for exploratory data analysis, cross-selling strategies, customer segmentation, and image recognition.

    - Applications of unsupervised learning

      - _News sections_
      - _Computer vision_
      - _Medical imaging_
      - _Anomaly detection_
      - _Customer personas_
      - _Recommendation engines_

    - Unsupervised learning approaches

      - Clustering

        - Clustering is a data mining technique which groups unlabeled data based on their similarities or differences. Clustering algorithms are used to process raw, unclassified data objects into groups represented by structures or patterns in the information. Clustering algorithms can be categorized into a few types, specifically exclusive, overlapping, hierarchical, and probabilistic.

          - [_Exclusive and Overlapping Clustering_](https://www.ibm.com/cloud/learn/unsupervised-learning#:~:text=Exclusive%20and%20Overlapping%20Clustering)

            - Exclusive clustering is a form of grouping that stipulates a data point can exist only in one cluster. This can also be referred to as “hard” clustering.
              - [K-means clustering algorithm](https://realpython.com/k-means-clustering-python/#understanding-the-k-means-algorithm) is an example of exclusive clustering.
            - Overlapping clusters differs from exclusive clustering in that it allows data points to belong to multiple clusters with separate degrees of membership. “Soft” or fuzzy k-means clustering is an example of overlapping clustering.

          - [_Hierarchical clustering - HCA_](https://www.ibm.com/cloud/learn/unsupervised-learning#:~:text=of%20overlapping%20clustering.-,Hierarchical%20clustering,-Hierarchical%20clustering%2C%20also)

          - [_Probabilistic clustering_](https://www.ibm.com/cloud/learn/unsupervised-learning#:~:text=of%20divisive%20clustering-,Probabilistic%20clustering,-A%20probabilistic%20model)

      - Association

        - An association rule is a rule-based method for finding relationships between variables in a given dataset. These methods are frequently used for market basket analysis, allowing companies to better understand relationships between different products. Understanding consumption habits of customers enables businesses to develop better cross-selling strategies and recommendation engines.
        - Examples of this can be seen in Amazon’s “Customers Who Bought This Item Also Bought” or Spotify’s "Discover Weekly" playlist. While there are a few different algorithms used to generate association rules, such as Apriori, Eclat, and FP-Growth, the Apriori algorithm is most widely used.

          - [_Apriori algorithms explained_](https://towardsdatascience.com/underrated-machine-learning-algorithms-apriori-1b1d7a8b7bc)
            - Apriori algorithm refers to the algorithm which is used to calculate the association rules between objects. It means how two or more objects are related to one another. In other words, we can say that the apriori algorithm is an association rule leaning that analyzes that people who bought product A also bought product B.
            - Apriori is an algorithm used for Association Rule Mining. It searches for a series of frequent sets of items in the datasets. It builds on associations and correlations between the itemsets. It is the algorithm behind “You may also like” where you commonly saw in recommendation platforms.

      - [Dimensionality reduction](https://machinelearningmastery.com/dimensionality-reduction-for-machine-learning/)

        - While more data generally yields more accurate results, it can also impact the performance of machine learning algorithms (e.g. overfitting) and it can also make it difficult to visualize datasets.
        - Dimensionality reduction is a technique used when the number of features, or dimensions, in a given dataset is too high. It reduces the number of data inputs to a manageable size while also preserving the integrity of the dataset as much as possible.
        - It is commonly used in the preprocessing data stage, and there are a few different dimensionality reduction methods that can be used, such as:

          - [_Principal component analysis - PCA with python_](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)

            - Principal component analysis (PCA) is a type of dimensionality reduction algorithm which is used to reduce redundancies and to compress datasets through feature extraction.
            - This method uses a linear transformation to create a new data representation, yielding a set of "principal components."
              - The first principal component is the direction which maximizes the variance of the dataset.
              - While the second principal component also finds the maximum variance in the data, it is completely uncorrelated to the first principal component, yielding a direction that is perpendicular, or orthogonal, to the first component.
              - This process repeats based on the number of dimensions, where a next principal component is the direction orthogonal to the prior components with the most variance.

          - [_Singular value decomposition SVD_](https://www.geeksforgeeks.org/singular-value-decomposition-svd/)

            - Singular value decomposition (SVD) is another dimensionality reduction approach which factorizes a matrix, A, into three, low-rank matrices.
            - SVD is denoted by the formula, A = USVT
              - Where U and V are orthogonal matrices
              - S is a diagonal matrix, and S values are considered singular values of matrix A.
            - Similar to PCA, it is commonly used to reduce noise and compress data, such as image files.

          - [_Autoencoders_](https://www.ibm.com/cloud/learn/unsupervised-learning#:~:text=as%20image%20files.-,Autoencoders,-Autoencoders%20leverage%20neural)
            - Autoencoders leverage neural networks to compress data and then recreate a new representation of the original data’s input.
            - The hidden layer specifically acts as a bottleneck to compress the input layer prior to reconstructing within the output layer.
            - The stage from the input layer to the hidden layer is referred to as “encoding” while the stage from the hidden layer to the output layer is known as “decoding.”

  - [Semi-supervised learning](https://www.analyticsvidhya.com/blog/2017/09/pseudo-labelling-semi-supervised-learning-technique/#h2_5)

    - Semi-supervised learning offers a happy medium between supervised and unsupervised learning. During training, it uses a smaller labeled data set to guide classification and feature extraction from a larger, unlabeled data set.
    - Semi-supervised learning can solve the problem of having not enough labeled data (or not being able to afford to label enough data) to train a supervised learning algorithm.
    - Semi-supervised learning uses pseudo labeling to train the model with less labeled training data than supervised learning.
    - The process can combine various neural network models and training ways. The whole working of semi-supervised learning is explained in the below points:
      - Firstly, it trains the model with less amount of training data similar to the supervised learning models. The training continues until the model gives accurate results.
      - The algorithms use the unlabeled dataset with pseudo labels in the next step, and now the result may not be accurate.
      - Now, the labels from labeled training data and pseudo labels data are linked together.
      - The input data in labeled training data and unlabeled training data are also linked.
      - In the end, again train the model with the new combined input as did in the first step. It will reduce errors and improve the accuracy of the model.

- [Reinforcement learning](https://www.ibm.com/cloud/learn/machine-learning#:~:text=What%27s%20the%20Difference%3F%22-,Reinforcement%20machine%20learning,-Reinforcement%20machine%20learning)

  - Reinforcement machine learning is a behavioral machine learning model that is similar to supervised learning, but the algorithm isn’t trained using sample data.
  - _This model learns as it goes by using trial and error_. A sequence of successful outcomes will be reinforced to develop the best recommendation or policy for a given problem.
  - [_Terms used in Reinforcement Learning_](https://www.javatpoint.com/reinforcement-learning#Terms)

    - _Agent()_: An entity that can perceive/explore the environment and act upon it.
    - _Environment()_: A situation in which an agent is present or surrounded by. In RL, we assume the stochastic environment, which means it is random in nature.
    - _Action()_: Actions are the moves taken by an agent within the environment.
    - _State()_: State is a situation returned by the environment after each action taken by the agent.
    - _Reward()_: A feedback returned to the agent from the environment to evaluate the action of the agent.
    - _Policy()_: Policy is a strategy applied by the agent for the next action based on the current state.
    - _Value()_: It is expected long-term retuned with the discount factor and opposite to the short-term reward.
    - _Q-value()_: It is mostly similar to the value, but it takes one additional parameter as a current action (a).

  - [Application](https://www.javatpoint.com/reinforcement-learning#Applications)

    - Robotics: RL is used in Robot navigation, Robo-soccer, walking, juggling, etc.
    - Control: RL can be used for adaptive control such as Factory processes, admission control in telecommunication, and Helicopter pilot is an example of reinforcement learning.
    - Game Playing: RL can be used in Game playing such as tic-tac-toe, chess, etc.
    - Chemistry: RL can be used for optimizing the chemical reactions.
    - Business: RL is now used for business strategy planning.
    - Manufacturing: In various automobile manufacturing companies, the robots use deep reinforcement learning to pick goods and put them in some containers.
    - Finance Sector: The RL is currently used in the finance sector for evaluating trading strategies.

  - [See overview about Reinforcement learning!](https://www.javatpoint.com/reinforcement-learning)
  - [Analytic Vidhya Blog](https://www.analyticsvidhya.com/blog/2021/02/introduction-to-reinforcement-learning-for-beginners/)
  - [Tutorial, Exams, Projects and Courses](https://neptune.ai/blog/best-reinforcement-learning-tutorials-examples-projects-and-courses)

- [Steps to Complete a Machine Learning Project](https://www.analyticsvidhya.com/blog/2021/04/steps-to-complete-a-machine-learning-project/)

  - Data collection
  - Data preparation
  - Train model on data in 3 steps:
    - Choose an algorithm
    - Overfit the model
    - Reduce overfitting with regularization
  - Analysis & Evalution
  - Serve model or Deploying a model
  - Retrain model

---

#### Linear regression

---

- [What is Linear regression?](https://www.ibm.com/topics/linear-regression)

  - Linear regression analysis is used to predict the value of a variable based on the value of another variable. The variable you want to predict is called the dependent variable. The variable you are using to predict the other variable's value is called the independent variable.

    - There are simple linear regression calculators that use a “least squares” method to discover the best-fit line for a set of paired data. You then estimate the value of X (dependent variable) from Y (independent variable)

  - Two type

    - Simple regression

      - Simple linear regression uses traditional slope-intercept form, where m and b are the variables our algorithm will try to “learn” to produce the most accurate predictions. x represents our input data and y represents our prediction.
      - `y = mx + b` or `Sales = Weight * Radio + Bias`

    - Multivariable regression

      - A more complex, multi-variable linear equation might look like this, where w represents the coefficients, or weights, our model will try to learn.
      - `f(x,y,z)= w1x + w2y + w3z` or `Sales = W1*Radio + W2*TV + W3*News`

    - [Gradient descent](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html#)

      - Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient.
      - In machine learning, we use gradient descent to update the parameters of our model. Parameters refer to coefficients in Linear Regression and weights in neural networks.
      - Example:

        ```text
        Gradient descent

            Bước 1: Khởi tạo giá trị x tùy ý
          Bước 2: Gán x = x – learning_rate * f'(x). Với learning_rate là hằng số không âm ví dụ learning_rate = 0.001
        Bước 3: Tính lại f(x):

        - Nếu f(x) đủ nhỏ thì dừng lại.
        - Ngược lại tiếp tục bước 2.
        ```

    - [Lost functions](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)

      - Use [Mean Square Error - MSE - L2](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#mse-l2) as cost function. MSE measures the average squared difference between an observation’s actual and predicted values.

    - See [more](https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html) about Linear regression!

    - Some types Lost functions

      - [Cross-entropy loss](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy) or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1.

      - [Mean Absolute Error - MAE - L1 loss](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#mae-l1)

      - [Mean Square Error - MSE - L2 loss](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#mse-l2)

      - Hinge
      - Huber
      - Kullback-Leibler

---

#### Logistic regression

---

- Content

---

#### Neural network

---

- Content

---

#### Convolutional neural network

---

- Content

---

#### CNN Techniques

---

- Content

---

#### Autoencoder

---

- Content

---

#### GAN

---

- Content

---

#### Object detection

---

- Content

---

#### RNN

---

- Content

---
