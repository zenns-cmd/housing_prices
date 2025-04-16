import math
import numpy as np
import matplotlib.pyplot as plt

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
    print("\n🚀 Starting gradient descent training...")
    for i in range(iterations):
        dw, db = computegradient(x, y, w, b)
        w = w - learningrate * dw
        b = b - learningrate * db
        currentcost = cost(x, y, w, b)
        costs.append(currentcost)
        if i % 100 == 0:
            print(f"🔄 Iteration {i}: Cost = {currentcost:.2f}, w = {w:.2f}, b = {b:.2f}")
    print("🎉 Training complete!\n")
    return w, b, costs

# data
x_train = np.array([1.0, 2.0]) 
y_train = np.array([300.0, 500.0])  

w = 0
b = 0  

learningrate = 0.1
iterations = 1000

w, b, costs = gradientdescent(x_train, y_train, w, b, learningrate, iterations)

plt.plot(range(iterations), costs)
plt.title("Cost vs. Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

plt.scatter(x_train, y_train, marker='x', c='purple', label='Training data')
tmp_fwb = prediction(x_train, w, b)
plt.plot(x_train, tmp_fwb, c='pink', label='Prediction')
plt.title("Housing Prices")
plt.ylabel("Price (in 1000s of dollars)")
plt.xlabel("Size (1000 sqft)")
plt.legend()
plt.show()

try:
    print("\n🔮 Let's predict a house price!")
    size = float(input("🏡 Enter the size of the house (in 1000 sqft): "))
    if size < min(x_train) or size > max(x_train):
        print("⚠️ Warning: The input is outside the range of the training data. Predictions may be less accurate.")
   
    prediction2 = prediction(size,round(w),round(b))
   
    cost2 = cost(x_train, y_train, w, b)
    print(f"\n📊 Model performance:")
    print(f"Cost (Mean Squared Error) for w = {round(w)}, b = {round(b)}: {round(cost2)}")
    plt.scatter(size,prediction2, marker='o', s=100, c='red', label='Your house')
    plt.scatter(x_train, y_train, marker='x', c='purple', label='Training data')
    plt.plot(x_train, tmp_fwb, c='pink', label='Prediction')
    plt.title(f"Predicted Price: ${prediction2 * 1000:,.2f}")
    plt.ylabel("Price (in 1000s of dollars)")
    plt.xlabel("Size (1000 sqft)")
    plt.legend()
    plt.show()
    print(f"\n💲 Predicted price for a {size} (1000 sqft) house: ${prediction2 * 1000:,.2f}")
    
    
except ValueError:
    print("❌ Invalid input, please enter a numeric value")


