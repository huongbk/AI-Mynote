# AI Note

---

## Deep Learning Basic

- Sumary content from Deep learning basic course [AI4E](https://www.facebook.com/nttuan8.AI4E/)

---

- [x] [Introduce ML](#introduce-ml)

- [x] [Linear regression](#linear-regression)

- [x] [Logistic regression](#logistic-regression)

- [x] [Neural network](#neural-network)

- [x] [Convolutional neural network](#convolutional-neural-network)

- [x] [CNN Techniques](#cnn-techniques)

- [x] [Autoencoder](#autoencoder)

- [ ] [GAN](#gan)

- [ ] [Object detection](#object-detection)

- [ ] [RNN](#rnn)

---

### Introduce ML

---

- [**What is Artificial Intelligent?**](https://www.ibm.com/cloud/learn/what-is-artificial-intelligence#:~:text=the%20human%20mind.-,What%20is%20artificial%20intelligence%3F,-While%20a%20number)

  - It is the science and engineering of making intelligent machines, especially intelligent computer programs. It is related to the similar task of using computers to understand human intelligence, but AI does not have to confine itself to methods that are biologically observable.
  - History of Artificial Intelligent: [_Key dates and names_](https://www.ibm.com/cloud/learn/what-is-artificial-intelligence#toc-history-of--2jPgsXm)

- [**What is Machine learning?**](https://www.ibm.com/cloud/learn/machine-learning#:~:text=within%20businesses%20today.-,What%20is%20machine%20learning%3F,-Machine%20learning%20is)

  - Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.

- [**How machine learning works?**](https://www.ibm.com/cloud/learn/machine-learning#toc-how-machin-NoVMSZI_)

  - UC Berkeley (link resides outside IBM) breaks out the learning system of a machine learning algorithm into three main parts.
    - A Decision Process: In general, machine learning algorithms are used to make a prediction or classification. Based on some input data, which can be labelled or unlabeled, your algorithm will produce an estimate about a pattern in the data.
    - An Error Function: An error function serves to evaluate the prediction of the model. If there are known examples, an error function can make a comparison to assess the accuracy of the model.
    - An Model Optimization Process: If the model can fit better to the data points in the training set, then weights are adjusted to reduce the discrepancy between the known example and the model estimate. The algorithm will repeat this evaluate and optimize process, updating weights autonomously until a threshold of accuracy has been met.

- **Machine learning methods** [IBM Blog](https://www.ibm.com/cloud/learn/artificial-intelligence)

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
        - Each node is made up of inputs, weights, a bias (or threshold), and an output. If that output value exceeds a given threshold, it ???fires??? or activates the node, passing data to the next layer in the network. Neural networks learn this mapping function through supervised learning, adjusting based on the loss function through the process of gradient descent.
        - When the cost function is at or near zero, we can be confident in the model???s accuracy to yield the correct answer.

      - _Navie bayes_

        - Naive Bayes is classification approach that adopts the principle of class conditional independence from the Bayes Theorem.
        - This means that the presence of one feature does not impact the presence of another in the probability of a given outcome, and each predictor has an equal effect on that result.
        - There are three types of Na??ve Bayes classifiers: Multinomial Na??ve Bayes, Bernoulli Na??ve Bayes, and Gaussian Na??ve Bayes. This technique is primarily used in text classification, spam identification, and recommendation systems.

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

            - Exclusive clustering is a form of grouping that stipulates a data point can exist only in one cluster. This can also be referred to as ???hard??? clustering.
              - [K-means clustering algorithm](https://realpython.com/k-means-clustering-python/#understanding-the-k-means-algorithm) is an example of exclusive clustering.
            - Overlapping clusters differs from exclusive clustering in that it allows data points to belong to multiple clusters with separate degrees of membership. ???Soft??? or fuzzy k-means clustering is an example of overlapping clustering.

          - [_Hierarchical clustering - HCA_](https://www.ibm.com/cloud/learn/unsupervised-learning#:~:text=of%20overlapping%20clustering.-,Hierarchical%20clustering,-Hierarchical%20clustering%2C%20also)

          - [_Probabilistic clustering_](https://www.ibm.com/cloud/learn/unsupervised-learning#:~:text=of%20divisive%20clustering-,Probabilistic%20clustering,-A%20probabilistic%20model)

      - Association

        - An association rule is a rule-based method for finding relationships between variables in a given dataset. These methods are frequently used for market basket analysis, allowing companies to better understand relationships between different products. Understanding consumption habits of customers enables businesses to develop better cross-selling strategies and recommendation engines.
        - Examples of this can be seen in Amazon???s ???Customers Who Bought This Item Also Bought??? or Spotify???s "Discover Weekly" playlist. While there are a few different algorithms used to generate association rules, such as Apriori, Eclat, and FP-Growth, the Apriori algorithm is most widely used.

          - [_Apriori algorithms explained_](https://towardsdatascience.com/underrated-machine-learning-algorithms-apriori-1b1d7a8b7bc)
            - Apriori algorithm refers to the algorithm which is used to calculate the association rules between objects. It means how two or more objects are related to one another. In other words, we can say that the apriori algorithm is an association rule leaning that analyzes that people who bought product A also bought product B.
            - Apriori is an algorithm used for Association Rule Mining. It searches for a series of frequent sets of items in the datasets. It builds on associations and correlations between the itemsets. It is the algorithm behind ???You may also like??? where you commonly saw in recommendation platforms.

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
            - Autoencoders leverage neural networks to compress data and then recreate a new representation of the original data???s input.
            - The hidden layer specifically acts as a bottleneck to compress the input layer prior to reconstructing within the output layer.
            - The stage from the input layer to the hidden layer is referred to as ???encoding??? while the stage from the hidden layer to the output layer is known as ???decoding.???

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

- [**Reinforcement learning**](https://www.ibm.com/cloud/learn/machine-learning#:~:text=What%27s%20the%20Difference%3F%22-,Reinforcement%20machine%20learning,-Reinforcement%20machine%20learning)

  - Reinforcement machine learning is a behavioral machine learning model that is similar to supervised learning, but the algorithm isn???t trained using sample data.
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

- [**Steps to Complete a Machine Learning Project**](https://www.analyticsvidhya.com/blog/2021/04/steps-to-complete-a-machine-learning-project/)

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

### Linear regression

---

- [**What is Linear regression?** IBM](https://www.ibm.com/topics/linear-regression)

  - Linear regression analysis is used to predict the value of a variable based on the value of another variable. The variable you want to predict is called the dependent variable. The variable you are using to predict the other variable's value is called the independent variable.

    - There are simple linear regression calculators that use a ???least squares??? method to discover the best-fit line for a set of paired data. You then estimate the value of X (dependent variable) from Y (independent variable)

  - Two type

    - Simple regression

      - Simple linear regression uses traditional slope-intercept form, where m and b are the variables our algorithm will try to ???learn??? to produce the most accurate predictions. x represents our input data and y represents our prediction.
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

            B?????c 1: Kh???i t???o gi?? tr??? x t??y ??
          B?????c 2: G??n x = x ??? learning_rate * f'(x). V???i learning_rate l?? h???ng s??? kh??ng ??m v?? d??? learning_rate = 0.001
        B?????c 3: T??nh l???i f(x):

        - N???u f(x) ????? nh??? th?? d???ng l???i.
        - Ng?????c l???i ti???p t???c b?????c 2.
        ```

    - [Lost functions](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)

      - Use [Mean Square Error - MSE - L2](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#mse-l2) as cost function. MSE measures the average squared difference between an observation???s actual and predicted values.

    - See [more](https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html) about Linear regression!

    - Some types Lost functions

      - [Cross-entropy loss](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy) or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1.

      - [Mean Absolute Error - MAE - L1 loss](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#mae-l1)

      - [Mean Square Error - MSE - L2 loss](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#mse-l2)

      - Hinge
      - Huber
      - Kullback-Leibler

---

### Logistic regression

---

- [**What is Logistic regression?**](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html)

  - Logistic regression is a classification algorithm used to assign observations to a discrete set of classes.
  - Unlike linear regression which outputs continuous number values, logistic regression transforms its output using the logistic sigmoid function to return a probability value which can then be mapped to two or more discrete classes.

- **Types**

  - Binary (Pass/Fail)
  - Multi (Cats, Dogs, Sheep)
  - Ordinal (Low, Medium, High)

- **Binary classification**

  - Pass/Fail
  - Activation function
    - [_Sigmoid activation_](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#sigmoid-activation)
      - In order to map predicted values to probabilities, we use the sigmoid function.
      - The function maps any real value into another value between 0 and 1.
  - [Decision boundary](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#decision-boundary)
    - Our current prediction function returns a probability score between 0 and 1.
    - In order to map this to a discrete class (true/false, cat/dog), we select a threshold value or tipping point _above_ which we will classify values into class 1 and _below_ which we classify values into class 2.
  - [Make prediction](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#making-predictions)

    - Using our knowledge of sigmoid functions and decision boundaries, we can now write a prediction function.
    - A prediction function in logistic regression returns the probability of our observation being positive, "True" or ???Yes???.
    - [_Math & Predict Function_](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#:~:text=in%20class%201.-,Math,-Let%E2%80%99s%20use%20the)

  - [Cost function](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#cost-function)

    - [_Binary Cross-entropy loss_](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)
    - [_Math_](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#:~:text=optimal%20global%20minimum.-,Math,-Instead%20of%20Mean)

  - Gradient descent
    - To minimize our cost, we use Gradient Descent just like before in Linear Regression.
    - [_Math & Code_](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#gradient-descent)

- [**Multclass Logistic Regression**](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#multiclass-logistic-regression)

  - Instead of y=0,1 we will expand our definition so that y=0,1...n.
  - Basically we re-run binary classification multiple times, once for each class.

  - Procedure

    - Divide the problem into `n+1` binary classification problems (+1 because the index starts at 0?).
    - For each class???
    - Predict the probability the observations are in that single class.
    - prediction = <math>max(probability of the classes)

  - [Andrew Ng explained](https://youtu.be/-EIfb6vFJzc?t=298)

---

### Neural network

---

- [**What is Neural network?**](https://ml-cheatsheet.readthedocs.io/en/latest/nn_concepts.html#neural-network)

  - Neural networks are a class of machine learning algorithms used to model complex patterns in datasets using _multiple hidden layers_ and _non-linear activation functions_. A neural network takes an input, passes it through multiple layers of hidden neurons (mini-functions with unique coefficients that must be learned), and outputs a prediction representing the combined input of all the neurons.

    - Neural networks are trained iteratively using optimization techniques like gradient descent.
    - After each cycle of training, an error metric is calculated based on the difference between prediction and target.
      - The derivatives of this error metric are calculated and propagated back through the network using a technique called backpropagation.
      - Each neuron???s coefficients (weights) are then adjusted relative to how much they contributed to the total error.
    - This process is repeated iteratively until the network error drops below an acceptable threshold.

  - [IBM overview](https://www.ibm.com/cloud/learn/neural-networks)

- **Neural**

  - A neuron takes a group of weighted inputs, applies an activation function, and returns an output.
    - The neuron then applies an activation function to the ???sum of weighted inputs??? from each incoming synapse and passes the result on to all the neurons in the next layer.

- **Synapse** _- Kh???p th???n kinh_

  - Synapses are like roads in a neural network.
  - Synapses are connect inputs to neurons, neurons to neurons, and neurons to outputs.
    - In order to get from one neuron to another, you have to travel along the synapse paying the ???toll??? (weight) along the way.
      - Each connection between two neurons has a _unique synapse_ with a _unique weight_ attached to it.
      - When we talk about updating weights in a network, we???re really talking about adjusting _the weights on these synapses_.

- **Weights**

  - Weights are values that control the strength of the connection between two neurons. That is, inputs are typically multiplied by weights, and that defines how much influence the input will have on the output.
    - In other words: when the inputs are transmitted between neurons, the weights are applied to the inputs along with an additional value (the `bias`)

- **Bias**

  - Bias terms are additional constants attached to neurons and added to the weighted input before the activation function is applied. Bias terms help models represent patterns that do not necessarily pass through the origin.
    - Bias in Neural Networks can be thought of as analogous to the role of a constant in a linear function, whereby the line is effectively transposed by the constant value.

- **Layers**

  - Input layer

    - Holds the data your model will train on.
    - Each neuron in the input layer represents a unique attribute in your dataset (e.g. height, hair color, etc.).

  - Hidden layer

    - Sits between the input and output layers and applies an activation function before passing on the results. There are often multiple hidden layers in a network.
    - In traditional networks, hidden layers are typically `fully-connected layers` each neuron receives input from all the previous layer???s neurons and sends its output to every neuron in the next layer.

  - Output layer
    - The final layer in a network. It receives input from the previous hidden layer, optionally applies an activation function, and returns an `output` representing your model???s prediction.

- **Weighted Input**

  - `A neuron???s input equals the sum of weighted outputs from all neurons in the previous layer`. Each input is multiplied by the weight associated with the synapse connecting the input to the current neuron.
    - If there are 3 inputs or neurons in the previous layer, each neuron in the current layer will have 3 distinct weights one for each each synapse.
  - Notice, it???s exactly the `same equation we use with linear regression!`

    - In fact, a neural network with a single neuron is the same as linear regression! The only `difference` is the neural network post-processes the weighted input with an activation function.

    ![Neural network!](https://www.researchgate.net/profile/Adem-Tekerek-2/publication/335903047/figure/fig1/AS:805611931443200@1569084260672/Proposed-artificial-neural-network-model-3-10-1-Neural-network-weight-vectors-are.png)

- **Activation Functions**

  - Activation functions live inside neural network layers and modify the data they receive before passing it to the next layer.
  - Activation functions give neural networks their power - allowing them to model complex non-linear relationships.

    - By modifying inputs with non-linear functions neural networks can model highly complex relationships between features.
    - Popular activation functions include [Relu](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#activation-relu) and [Sigmoid](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#sigmoid).

  - Activation functions typically have the following properties:
    - Non-linear
    - Continuously differentiable
    - Fixed Range

- **Loss Functions**

  - A loss function, or cost function, is a wrapper around our model???s predict function that tells us ???how good??? the model is at making predictions for a given set of parameters.
  - The loss function has its own curve and its own derivatives. The slope of this curve tells us how to change our parameters to make the model more accurate!

    - We use the model to make predictions.
    - We use the cost function to update our parameters.
    - Our cost function can take a variety of forms as there are many different cost functions available.

  - Popular loss functions include: [MSE (L2)](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#mse) and [Cross-entropy Loss](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy).
    - MSE or Mean Squared Error or L2 loss
    - Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1.

- **Optimization Algorithms**

  - Optimizers are algorithms or methods used to change the _attributes_ of your neural network such as `weights` and `learning rate` in order to `reduce the losses`.

    - How you should change your weights or learning rates of your neural network to reduce the losses is defined by the optimizers you use.
    - Optimization algorithms or strategies are responsible for `reducing the losses` and to provide the `most accurate` results possible.

  - Some [optimization algorithms](https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6) in neural network as below

    - _Gradient Descent_
    - _Stochastic Gradient Descent_
    - _Mini-Batch Gradient Descent_
    - _Momentum_
    - _Nesterov Accelerated Gradient_
    - _Adagrad_
    - _AdaDelta_
    - _Adam_

      - `Adam` is the best optimizers if you wants to train the neural network in less time and more efficiently
      - If you want to use `Gradient descent` algorithm than `Mini-batch Gradient descent` is the best option.

- **Gradient Accumulation**

  - Gradient accumulation is a mechanism to `split` the `batch` of samples - used for training a neural network into several `mini-batches` of samples that will be run sequentially.

  - When do you need Gradient accumulation?

    - This is used to enable using large batch sizes that `require more GPU memory than available.`
    - Gradient accumulation helps in doing so by `using mini-batches` that require an amount of GPU memory that can be satisfied.

  - How it works?

    - Gradient accumulation means `running all mini-batches sequentially` (generally on the same GPU) `while accumulating their calculated gradients and not updating the model variables` - the weights and biases of the model.
    - The model variables must not be updated `during` the accumulation in order to ensure all mini-batches use the same model variable values to calculate their gradients.
    - Only `after` accumulating the gradients of all those mini-batches will we generate and apply the updates for the model variables.

    ![Gradient accumulation](https://ml-cheatsheet.readthedocs.io/en/latest/_images/gradient_accumulation.png)

---

### Convolutional neural network

---

- **Underfitting/Overfitting**

  - [_Underfitting_](https://www.ibm.com/cloud/learn/underfitting)

    - Underfitting is a scenario in data science where a data model is unable to capture the relationship between the input and output variables accurately, generating a high `error rate on both the training set and unseen data.`
    - It occurs when a model is too simple, which can be a result of a model `needing more training time`, `more input features`, or `less regularization`.
    - How to avoid underfitting?
      - `Increase model complexity.`
      - `Increase the number of features, performing feature engineering.`
      - `Remove noise from the data.`
      - `Increase the number of epochs or increase the duration of training to get better results.`

  - [_Overfitting_](https://www.ibm.com/cloud/learn/overfitting)

    - Overfitting is a concept in data science, which occurs when a statistical model fits exactly against its training data.
    - When this happens, the algorithm unfortunately cannot perform accurately against unseen data, defeating its purpose.
    - How to avoid overfitting?

      - `Early stopping`: As we mentioned earlier, this method seeks to pause training before the model starts learning the noise within the model.
      - `Train with more data`: Otherwise, you could just continue to add more complexity to the model, causing it to overfit.
      - `Data augmentation`: While it is better to inject clean, relevant data into your training data, sometimes noisy data is added to make a model more stable. However, this method should be done sparingly.
      - `Feature selection`: Feature selection is the process of identifying the most important ones within the training data and then eliminating the irrelevant or redundant ones. This is commonly mistaken for dimensionality reduction, but it is different. However, both methods help to simplify your model to establish the dominant trend in the data.
      - [`Regularization`](https://ml-cheatsheet.readthedocs.io/en/latest/regularization.html): If overfitting occurs when a model is too complex, it makes sense for us to reduce the number of features.

        - But what if we don???t know which inputs to eliminate during the feature selection process? If we don???t know which features to remove from our model, regularization methods can be particularly helpful.

          - Regularization applies a ???penalty??? to the input parameters with the larger coefficients, which subsequently limits the amount of variance in the model.
          - While there are a number of regularization methods, such as:

            - [`Data Augmentation`](https://ml-cheatsheet.readthedocs.io/en/latest/regularization.html#data-augmentation)
            - [`Dropout`](https://ml-cheatsheet.readthedocs.io/en/latest/regularization.html#dropout)
            - [`Early Stopping`](https://ml-cheatsheet.readthedocs.io/en/latest/regularization.html#early-stopping)
            - [`Ensembling`](https://ml-cheatsheet.readthedocs.io/en/latest/regularization.html#ensembling)
            - [`Injecting Noise`](https://ml-cheatsheet.readthedocs.io/en/latest/regularization.html#injecting-noise)
            - [`L1 Regularization`](https://ml-cheatsheet.readthedocs.io/en/latest/regularization.html#l1-regularization) or [`Lasso regularization`](https://ml-cheatsheet.readthedocs.io/en/latest/regularization.html#l1-regularization)
            - [`L2 Regularization`](https://ml-cheatsheet.readthedocs.io/en/latest/regularization.html#l2-regularization)

          - _They all seek to identify and reduce the noise within the data._

      - `Ensemble methods`: Ensemble learning methods are made up of a set of classifiers `e.g. decision trees` and their predictions are aggregated to identify the most popular result.

        - The most well-known ensemble methods are [`bagging and boosting`](https://www.geeksforgeeks.org/bagging-vs-boosting-in-machine-learning/).
          - `Bagging`
            - It is a homogeneous weak learners??? model that learns from each other independently in parallel and combines them for determining the model average.
              ![Bagging](https://media.geeksforgeeks.org/wp-content/uploads/20210707140912/Bagging.png)
          - `Bootsting`
            - It is also a homogeneous weak learners??? model but works differently from Bagging. In this model, learners learn sequentially and adaptively to improve model predictions of a learning algorithm.
              ![Bootsting](https://media.geeksforgeeks.org/wp-content/uploads/20210707140911/Boosting.png)

- **What problem with Neural Network?**

  - Blackbox
  - Quality predicts need large amount of data
  - Computationally expensive, requires hight processing time for large neural network
  - Hardware Dependence

- **Image processing**

  - [Kernel](<https://en.wikipedia.org/wiki/Kernel_(image_processing)>)

    - In image processing, a kernel, convolution matrix, or mask is a small matrix used for blurring, sharpening, embossing, edge detection, and more. This is accomplished by doing a convolution between the kernel and an image.

  - [RGB Image](http://spsprashanth.blogspot.com/2016/08/rgb.html)

    - RGB represents Red, Green and Blue. It is an additive color model. This three colors(RGB) can mix together to form any color.
    - The main purpose of the RGB color model is for the sensing, representation and display of images in electronic systems, such as televisions and computers.
    - Each level in RGB is represented by the range of decimal numbers from 0 to 255.
    - The total number of available colors is 256 x 256 x 256 or 16,777,216 possible colors.
    - In HTML, the color for a page background or text font is specified by an RGB value, expressed with six digits in hexadecimal format as shown below:

    ![RGB](https://4.bp.blogspot.com/-cY2aQB_7tRs/V8WVV87gjcI/AAAAAAAAAkM/S2lmFMe6ULI5TGeGV76MDmXc_YMKW_AWgCEw/s640/ColorImageRGB.gif)

    ![Image RGB](https://media.geeksforgeeks.org/wp-content/uploads/Pixel.jpg)

- **Convolutional operation**

  - [`Convolution layer (CONV)`](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)

    - The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the `input I` with respect to its dimensions. Its hyperparameters include the `filter F` and `stride S`. The resulting `output O` is called feature map or activation map.
    - [CS231n Stanford](https://cs231n.github.io/convolutional-networks/)

  - `Pooling layer`

    - The pooling layer (POOL) is a `downsampling operation`, typically applied after a convolution layer, which does some spatial invariance.
    - In particular, `max pooling` and `average pooling` are special kinds of pooling where the maximum and average value is taken, respectively.

  - `Fully Connected (FC)`

    - The fully connected layer (FC) operates on a `flattened` input where each input is connected to all neurons. If present, `FC layers` are usually found towards the end of CNN architectures and can be used to optimize objectives such as class scores.
      ![Plattened to FC](https://stanford.edu/~shervine/teaching/cs-230/illustrations/fully-connected-ltr.png?32caf9e07c79d652faa292812579d063)

  - [`Filter hyperparameters`](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks#filter)

    - In machine learning, a `hyperparameter` is a parameter whose value is used to control the learning process.

      - _Dimensions_ of a filter - s??? l???p filter ch???ng l??n nhau
      - _Stride_ - b?????c nh???y
      - _Padding_ - l???p ?????m zero b??n ngo??i

    - [_Tuning hyperparameter see here!_](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks#hyperparameters)

      - `I` is the length of the input volume size,
      - `F` is the length of the filter
      - `P` is the amount of zero padding
      - `S` is the stride, then the output size `O` of the feature map along that dimension is given by function: `O = (I - F + P_start + P_end)/S + 1`
        ![Function](https://stanford.edu/~shervine/teaching/cs-230/illustrations/parameter-compatibility-en.jpeg?bc91caf0473dc42f1a495946f67726d3)

- **Building blocks (Convolutional layer, pooling layer)**

  ![CNN](https://editor.analyticsvidhya.com/uploads/90650dnn2.jpeg)

- **ImageNet challenge**

  - [ILSVRC - The ImageNet Large Scale Visual Recognition Challenge](https://www.image-net.org/challenges/LSVRC/) is a benchmark in object category classification and detection on hundreds of object categories and millions of images. The challenge has been run annually from 2010 to present, attracting participation from more than fifty institutions.

- **VGG 16**

  ![VGG16](https://www.researchgate.net/profile/Max-Ferguson/publication/322512435/figure/fig3/AS:697390994567179@1543282378794/Fig-A1-The-standard-VGG-16-network-architecture-as-proposed-in-32-Note-that-only.png)

- **CNN applications**

  - Image Classification
  - Search Engines
  - Recommender Systems
  - Social Media
  - Medical Image Computing
  - Healthcare Data Science / Predictive Analytics
  - Etc.

---

### CNN Techniques

---

- [**Forward propagation**](https://ml-cheatsheet.readthedocs.io/en/latest/forwardpropagation.html#)

  - Forward propagation is how neural networks make predictions. Input data is ???forward propagated??? through the network layer by layer to the final layer which outputs a prediction.

- **Backpropagation**

  - The goals of backpropagation are straightforward: adjust each weight in the network in proportion to how much it contributes to overall error. If we iteratively reduce each weight???s error, eventually we???ll have a series of weights that produce good predictions.

  - Compare

    - _Forward Propagation_: Receive input data, process the information, and generate output.
    - _Backward Propagation_: Calculate error and update the parameters of the network.

  - [The Backpropagation Algorithm](https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd)

    - Backpropagation algorithm is probably the most fundamental building block in a neural network. It was first introduced in 1960s and almost 30 years later (1989) popularized by Rumelhart, Hinton and Williams in a paper called ???Learning representations by back-propagating errors???.
    - The algorithm is used to effectively train a neural network through a method called chain rule. In simple terms, after each forward pass through a network, backpropagation performs a backward pass while adjusting the model???s parameters (weights and biases).
    - T??m l???i:

      - T??nh ?????o h??m c???a Loss function v???i m???i h??? s??? `Weight`, `bias` b???ng chainrule r???i ??i???u ch???nh tr???ng s??? theo

    - Recommendation:
      - [_Chain rule refresher_](https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html#chain-rule-refresher)
      - [_Backpropagation Tuan Nguyen_](https://nttuan8.com/bai-4-backpropagation/)
      - [_NLP Processing with Deep learning_](https://www.youtube.com/watch?v=isPiE-DBagM&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6&ab_channel=StanfordUniversitySchoolofEngineering)

- [**Vanishing and Exploding Gradients in Neural Networks**](https://www.numpyninja.com/post/vanishing-and-exploding-gradients-in-neural-networks)

  - What is gradient?

    - The Gradient is nothing but a derivative of loss function with respect to the weights. It is used to updates the weights to minimize the loss function during the back propagation in neural networks.

  - What is Vanishing Gradients?

    - Vanishing Gradient occurs when the derivative or slope will get `smaller and smaller` as we go backward with every layer during backpropagation.
    - When weights update is very small or exponential small, the training time takes too much longer, and in the worst case, this _may completely stop the neural network training._
      - A vanishing Gradient problem _occurs with the sigmoid and tanh activation_ function because the derivatives of the sigmoid and tanh activation functions are between 0 to 0.25 and 0???1.
      - Therefore, the updated weight values are small, and the new weight values are very similar to the old weight values. This leads to Vanishing Gradient problem.
    - We can avoid this problem using the ReLU activation function because the gradient is 0 for negatives and zero input, and 1 for positive input. ( or _Leaky Relu_)

  - What is Exploding Gradients?

    - Exploding gradient occurs when the derivatives or slope will get `bigger and bigger` as we go backward with every layer during backpropagation. This situation is the exact opposite of the vanishing gradients.

      - This problem happens _because of weights, not because of the activation function_. Due to high weight values, the derivatives will also higher so that the new weight varies a lot to the older weight, and the gradient will never converge.
      - So it may result in oscillating around minima and never come to a global minima point.

    - [How to identify exploding gradients?](https://analyticsindiamag.com/complete-guide-to-exploding-gradient-problem/#:~:text=How%20to%20identify%20exploding%20gradients%3F)

      - If the model weights become unexpectedly large in the end.
      - Your model has a poor loss.
      - Or the model displays NaN loss whilst training.
      - The gradient value for error persists over 1.0 for every subsequent iteration during training.

    - [How to deal with an exploding gradient?](https://analyticsindiamag.com/complete-guide-to-exploding-gradient-problem/#:~:text=How%20to%20deal%20with%20an%20exploding%20gradient%3F)
      - Re-Design the Network Model
      - Use LSTM???s (Long short term memory)
      - Gradient Clipping
      - Weight Regularization

- **Resnet introduction**

  - `ResNet` is a Convolutional Neural Network (CNN) architecture that `overcame the ???vanishing gradient???` problem, making it possible to construct networks with up to thousands of convolutional layers, which `outperform shallower networks.`

  - Why resnet works?

    - Here, we have something called Residual blocks.
    - Many `Residual blocks` are stacked together to form a ResNet.
    - We have `Skipped connections` which are the major part of ResNet.
    - The following image below was provided by the authors in the original paper which denotes how a residual network works.

      - The idea is to connect the input of a layer directly to the output of a layer after skipping a few connections.
      - We can see here, `x` is the input to the layer which we are directly using to connect to a layer after skipping the identity connections and if we think the output from identity connection to be `F(x)`.
      - Then we can say the output will be `F(x) + x`.

      <img src="https://neurohive.io/wp-content/uploads/2019/01/resnet-e1548261477164.png" width="60%">

  - Paper [_Deep Residual Learning for Image Recognition_](https://arxiv.org/pdf/1512.03385.pdf)

- **Transfer learning**

  - [What is Transfer learning?](https://developer.ibm.com/articles/transfer-learning-for-deep-learning/)

    - Transfer learning is the method of starting with a pre-trained model and training it for a new ??? related ??? problem domain. The pre-trained network serves as transferred knowledge to be applied in another domain.

  - Methods for transfer learning

    - [_Feature transfer_](<https://developer.ibm.com/articles/transfer-learning-for-deep-learning/#:~:text=The%20idea%20behind%20feature%20transfer%20is%20then%20to%20use%20the%20input%20and%20feature%2Dextraction%20layers%20that%20have%20been%20trained%20with%20a%20given%20data%20set%20(with%20their%20weights%20and%20structure%20frozen)%20and%20train%20a%20new%20classification%20layer%20for%20the%20related%20problem%20domain>)

      - The idea behind feature transfer is then to use the input and feature-extraction layers that have been trained with a given data set (with their weights and structure frozen) and train a new classification layer for the related problem domain.

        <img src="https://developer.ibm.com/developer/default/articles/transfer-learning-for-deep-learning/images/Figure3.png">

    - [_Fine-tuning_](https://developer.ibm.com/articles/transfer-learning-for-deep-learning/#:~:text=domains%20are%20similar.-,Fine%2Dtuning,-In%20the%20prior)

      - This fine-tuning could mean that we train the later layers of the deep learning network (meaning adjust the weights based upon the classification error) while leaving the earlier layers frozen.
        <img src="https://developer.ibm.com/developer/default/articles/transfer-learning-for-deep-learning/images/Figure4.png">

    - [_Using a pre-trained model_](https://developer.ibm.com/articles/transfer-learning-for-deep-learning/#:~:text=to%20be%20classified.-,Using%20a%20pre%2Dtrained%20model,-With%20feature%20transfer)

      - Using a pre-trained model, the model can be refined through feature transfer (train the classification layer) or by freezing some of the early convolutional layers and retraining the later ones.
      - Given that the early convolutional layers detect general features that can be independent of the problem, retraining (or fine-tuning) the later convolutional layers where features are more specific and dependent on the problem can be useful.

  - When use Transfer learning?
    <img src="https://cdn.hackernoon.com/hn-images/1*xGpYptYPEqGl6gWr6bHZEQ.png" width="60%">

    - [See more here!](https://phamdinhkhanh.github.io/2020/04/15/TransferLearning.html#6-kinh-nghi%E1%BB%87m-transfer-learning)

- **Data augmentation**

  - Data augmentation in data analysis are techniques used to increase the amount of data by adding slightly modified copies of already existing data or newly created synthetic data from existing data. It acts as a regularizer and helps reduce overfitting when training a machine learning model.
  - [_Top 13 Data Augmentation Techniques: Comprehensive Guide_](https://research.aimultiple.com/data-augmentation-techniques/)
  - [_What is Data Augmentation? Techniques, Benefit & Examples_](https://research.aimultiple.com/data-augmentation/)
    <img src="https://research.aimultiple.com/wp-content/uploads/2021/04/data-augmentation-techniques-800x450.png">

- **Data synthesis**

  - Synthetic data, as the name suggests, is data that is artificially created rather than being generated by actual events. It is often created with the help of algorithms and is used for a wide range of activities, including as test data for new products and tools, for model validation, and in AI model training.

  - [_The Ultimate Guide to Synthetic Data: Uses, Benefits & Tools_](https://research.aimultiple.com/synthetic-data/)

---

### Autoencoder

---

- **Unsupervised Learning**

  - As session `Unsupervised learning` in chapter [Introduce ML](#introduce-ml)

- **Autoencoder (AE)**

  - Autoencoder is a type of neural network that can be used to learn a compressed representation of raw data.
    - An autoencoder is composed of an encoder and a decoder sub-models
    - The autoencoder idea was a part of NN history for decades (LeCun et al, 1987).
    - Traditionally an autoencoder is used for `dimensionality reduction` and feature learning.
  - [See more here!](https://ml-cheatsheet.readthedocs.io/en/latest/architectures.html#autoencoder)
    <img src="https://cdn.hackernoon.com/hn-images/1*-5D-CBTusUnmsbA6VYdY3A.png" width="80%">

- **Autoencoder application**

  - `Image denoising`
  - `Image compression`
  - `Text Retrieval`
  - `Similar image search`

- **Upsampling Techniques**

  - [`See more here!`](https://towardsdatascience.com/transposed-convolution-demystified-84ca81b4baba)
    - `Nearest Neighbors`
    - `Bi-Linear Interpolation`
    - `Bed Of Nails`
    - `Max-Unpooling`
    - `Transposed Convolutions`

- **Convolutional Auto Encoder**

  - [`Explain Transposed convolution`](https://d2l.ai/chapter_computer-vision/transposed-conv.html)

  - `Stride = 1`

    <img src="https://d2l.ai/_images/trans_conv.svg">

  - `Stride = 2`

    <img src="https://d2l.ai/_images/trans_conv_stride2.svg">

  - Convolutional Auto Encoder

    <img src="https://www.researchgate.net/profile/Xifeng-Guo/publication/320658590/figure/fig1/AS:614154637418504@1523437284408/The-structure-of-proposed-Convolutional-AutoEncoders-CAE-for-MNIST-In-the-middle-there.png" width="80%">

  - [_Application Convolution AE_](https://iq.opengenus.org/applications-of-autoencoders/)

    - `Dimensionality Reduction`
    - `Image Compression`
    - `Image Denoising`
    - `Super Resolution`
    - `Semantic Segmentation`
    - `Feature Extraction`
    - `Image generation`
    - `Sequence to sequence prediction`
    - `Recommendation system`
    - `Etc.`

---

### GAN

---

- Content

---

### Object detection

---

- Content

---

### RNN

---

- Content

---
