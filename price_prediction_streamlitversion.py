import math
import numpy as np
import matplotlib.pyplot as plt



import streamlit as st 
import time

with st.spinner("Waking up the app... Please wait 30 seconds"):
    time.sleep(30) 
    


def prediction(x, w, b):
    
    f_wb = w * x + b
    return f_wb

def cost(x,y,w,b):
    m = len(x)
    cost = 0
    for i in range(m):
        f_wb = w*x[i] +b
        cost += (f_wb - y[i]) ** 2
    cost = cost/(2*m)
    return cost

def computegradient(x,y,w,b):
    m = len(x)
    dw = 0
    db = 0

    for i in range(m):
     fwb = w * x[i] + b
     dwtmp = (fwb - y[i])*x[i]
     dbtmp = (fwb-y[i])
     dw += dwtmp
     db += dbtmp
     dw = dw/m
     db = db/m
    return dw, db

def gradientdescent(x, y, w, b, learningrate, iterations):
    costs = []
    st.write("\n🚀 Starting gradient descent training...")
    for i in range(iterations):
        dw, db = computegradient(x, y, w, b)
        w = w - learningrate * dw
        b = b - learningrate * db
        currentcost = cost(x, y, w, b)
        costs.append(currentcost)
        if i % 100 == 0:
            st.write(f"🔄 Iteration {i}: Cost = {currentcost:.2f}, w = {w:.2f}, b = {b:.2f}")
    st.write("🎉 Training complete!\n")
    return w, b, costs


st.title("🏠 Housing Price Predictor")
st.subheader("Linear Regression with Gradient Descent")

# data
x_train = np.array([1.0, 2.0]) 
y_train = np.array([300.0, 500.0])  
w = 0
b = 0  
learningrate = 0.1
iterations = 1000


w, b, costs = gradientdescent(x_train, y_train, w, b, learningrate, iterations)

st.subheader("Training Progress")
fig1, ax1 = plt.subplots()
ax1.plot(range(iterations), costs)
ax1.set_title("Cost vs. Iterations")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Cost")
st.pyplot(fig1)


st.subheader("Regression Model")
fig2, ax2 = plt.subplots()
tmp_fwb = prediction(x_train, w, b)
ax2.scatter(x_train, y_train, marker='x', c='purple', label='Training data')
ax2.plot(x_train, tmp_fwb, c='pink', label='Prediction')
ax2.set_title("Housing Prices")
ax2.set_ylabel("Price (in 1000s of dollars)")
ax2.set_xlabel("Size (1000 sqft)")
ax2.legend()
st.pyplot(fig2)


st.subheader("🔮 Make a Prediction")
size = st.number_input("🏡 Enter the size of the house (in 1000 sqft):", min_value=0.1, max_value=10.0, value=1.5, step=0.1)

if size < min(x_train) or size > max(x_train):
    st.warning("⚠️ Warning: The input is outside the range of the training data. Predictions may be less accurate.")

prediction2 = prediction(size,round(w),round(b))
cost2 = cost(x_train, y_train, w, b)

st.subheader("📊 Results")
st.write(f"Model parameters: w = {round(w)}, b = {round(b)}")
st.write(f"Cost (Mean Squared Error): {round(cost2)}")


fig3, ax3 = plt.subplots()
ax3.scatter(size, prediction2, marker='o', s=100, c='red', label='Your house')
ax3.scatter(x_train, y_train, marker='x', c='purple', label='Training data')
ax3.plot(x_train, tmp_fwb, c='pink', label='Prediction')
ax3.set_title(f"Predicted Price: ${prediction2 * 1000:,.2f}")
ax3.set_ylabel("Price (in 1000s of dollars)")
ax3.set_xlabel("Size (1000 sqft)")
ax3.legend()
st.pyplot(fig3)

st.success(f"💲 Predicted price for a {size} (1000 sqft) house: ${prediction2 * 1000:,.2f}")
