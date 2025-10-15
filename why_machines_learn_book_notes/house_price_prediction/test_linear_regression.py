import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from linear_regression_model import HousePriceModel

def main():
    # Create an instance of the housing price model
    model = HousePriceModel()
    
    # Generate synthetic housing data
    print("Generating synthetic housing data...")
    housing_data = model.generate_data(n_samples=200, seed=42)
    
    print("Data sample:")
    print(housing_data.head())
    
    print("\nData statistics:")
    print(housing_data.describe())
    
    # Plot the data to visualize relationships
    plot_data_relationships(housing_data)
    
    # Split the data into training and testing sets
    X = housing_data[['bedrooms', 'square_footage']]
    y = housing_data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Train the model
    print("\nTraining model...")
    model.train(X_train, y_train)
    
    # Get and display model parameters
    params = model.get_model_params()
    print("\nModel parameters:")
    print(f"Intercept: ${params['intercept']:.2f}")
    print(f"Coefficient for bedrooms: ${params['coefficients']['bedrooms']:.2f}")
    print(f"Coefficient for square footage: ${params['coefficients']['square_footage']:.2f}")
    
    print(f"\nThis means: Price = ${params['intercept']:.2f} + "
          f"${params['coefficients']['bedrooms']:.2f} * bedrooms + "
          f"${params['coefficients']['square_footage']:.2f} * square_footage")
    
    # Evaluate the model
    print("\nEvaluating model on test data...")
    eval_results = model.evaluate(X_test, y_test)
    
    print("\nModel evaluation:")
    print(f"Mean Squared Error: ${eval_results['mse']:.2f}")
    print(f"Root Mean Squared Error: ${eval_results['rmse']:.2f}")
    print(f"R² Score: {eval_results['r2']:.4f}")
    
    # Test with some example houses
    test_houses(model)
    
    # Plot actual vs predicted values
    plot_actual_vs_predicted(y_test, eval_results['predictions'])

def plot_data_relationships(housing_data):
    """Create visualizations of the housing data relationships."""
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
    print("Data visualization saved to 'housing_data_visualization.png'")

def test_houses(model):
    """Test the model with some example houses."""
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

def plot_actual_vs_predicted(y_test, y_pred):
    """Create a scatter plot of actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Actual vs Predicted House Prices')
    plt.xlabel('Actual Prices ($)')
    plt.ylabel('Predicted Prices ($)')
    plt.grid(True)
    plt.savefig('actual_vs_predicted.png')
    plt.close()
    print("Actual vs Predicted plot saved to 'actual_vs_predicted.png'")
    
    # Save actual vs predicted values in a DataFrame for easy viewing
    results_df = pd.DataFrame({
        'Actual Price': y_test.values,
        'Predicted Price': y_pred,
        'Difference': y_test.values - y_pred
    })
    print("\nSample of prediction results on test data:")
    print(results_df.head())

if __name__ == "__main__":
    main()