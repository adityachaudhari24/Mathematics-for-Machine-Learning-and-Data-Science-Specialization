import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic housing data
def generate_housing_data(n_samples=100):
    # Generate number of bedrooms (1-6)
    bedrooms = np.random.randint(1, 7, size=n_samples)
    
    # Generate square footage (600-3000 sq ft)
    # Make it somewhat correlated with number of bedrooms
    square_footage = 500 + bedrooms * 300 + np.random.normal(0, 200, size=n_samples)
    square_footage = np.clip(square_footage, 600, 3000).astype(int)
    
    # Generate prices with some noise
    # Base price: $100,000
    # Per bedroom: $50,000
    # Per sq ft: $100
    # Plus some noise
    price = (
        100000 +  # Base price
        50000 * bedrooms +  # Price per bedroom
        100 * square_footage +  # Price per square foot
        np.random.normal(0, 25000, size=n_samples)  # Random variation
    )
    
    # Create a DataFrame
    housing_data = pd.DataFrame({
        'bedrooms': bedrooms,
        'square_footage': square_footage,
        'price': price
    })
    
    return housing_data

# Generate and display some sample data
housing_data = generate_housing_data(200)
print("Sample of generated housing data:")
print(housing_data.head())

# Explore the data
print("\nData statistics:")
print(housing_data.describe())

# Plot the data to visualize relationships
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(housing_data['bedrooms'], housing_data['price'])
plt.title('Price vs. Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price ($)')

plt.subplot(1, 2, 2)
plt.scatter(housing_data['square_footage'], housing_data['price'])
plt.title('Price vs. Square Footage')
plt.xlabel('Square Footage')
plt.ylabel('Price ($)')

plt.tight_layout()
plt.savefig('housing_data_visualization.png')
plt.close()

# Split the data into training and testing sets
X = housing_data[['bedrooms', 'square_footage']]
y = housing_data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the model parameters
print("\nModel parameters:")
print(f"Intercept: ${model.intercept_:.2f}")
print(f"Coefficient for bedrooms: ${model.coef_[0]:.2f}")
print(f"Coefficient for square footage: ${model.coef_[1]:.2f}")
print(f"\nThis means: Price = ${model.intercept_:.2f} + "
      f"${model.coef_[0]:.2f} * bedrooms + ${model.coef_[1]:.2f} * square_footage")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel evaluation:")
print(f"Mean Squared Error: ${mse:.2f}")
print(f"Root Mean Squared Error: ${rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Let's test the model with some example houses
test_houses = [
    {'bedrooms': 3, 'square_footage': 1500},
    {'bedrooms': 2, 'square_footage': 1000},
    {'bedrooms': 4, 'square_footage': 2200},
    {'bedrooms': 5, 'square_footage': 2800}
]

print("\nPredictions for sample houses:")
for house in test_houses:
    bedrooms = house['bedrooms']
    square_footage = house['square_footage']
    price_pred = model.predict([[bedrooms, square_footage]])[0]
    print(f"House with {bedrooms} bedrooms and {square_footage} sq ft: "
          f"Predicted price = ${price_pred:.2f}")

# Save actual vs predicted values in a DataFrame for easy viewing
results_df = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred,
    'Difference': y_test - y_pred
})
print("\nSample of prediction results on test data:")
print(results_df.head())

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Prices ($)')
plt.ylabel('Predicted Prices ($)')
plt.grid(True)
plt.savefig('actual_vs_predicted.png')
plt.close()