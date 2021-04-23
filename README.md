The included Python file is the code I wrote to implement a convolutional neural network for sentiment analysis of movie reviews. After deriving the backward propagation expressions, I implemented the entire network using basic Python libraries and no standard machine learning packages. I've also included the training/test data and vocabulary I built my model on.

After getting the functionality down, I worked on optimizing my code. I used vectorizing operations (NumPy) and re-engineered the way I stored weights to achieve a **60x speedup for training the model.**

As a benchmark, I was able to achieve about 88% accuracy on test data, and that jumped up to 92% after k-fold cross validation and hyperparameter tuning.

I had a lot of fun implementing this model for a common NLP task, sentiment analysis, and definitely learned a lot about how important code runtime and scalability is.

(During the process, I coded mainly in a Jupyter notebook environment which is why some comments may appear out of place in the python file)
