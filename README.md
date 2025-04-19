This project predicts housing prices by applying gradient descent to a linear regression model built from scratch


✨ Interactive Version (using streamlit):

I've deployed an interactive version of the code at https://housingprices101.streamlit.app

I uploaded both the original and streamlit version of the code on this repository.
The files requirements.txt and psh.py are just extras used for deployment.


✅ Standard Algorithms:

- Hypothesis Function: prediction(x, w, b) computes the linear model output.

- Cost Function: cost(x, y, w, b) calculates Mean Squared Error.

- Gradient Descent: gradientdescent(x, y, w, b, learningrate, iterations) optimizes model parameters w and b.


📊 Visualizations:

- Cost vs. Iterations: Tracks how the cost decreases over training iterations.

- Regression Line: Plots the best fit line against training data.


💡 Prediction:

- Users can input a house size to get a predicted price.

- Warns if input is outside the training data range.

