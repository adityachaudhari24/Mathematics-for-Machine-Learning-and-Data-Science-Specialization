import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class HousePriceModel:
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
        
    def generate_data(self, n_samples=100, seed=42):
        """Generate synthetic housing data for training and testing."""
        np.random.seed(seed)
        
        # Generate number of bedrooms (1-6)
        bedrooms = np.random.randint(1, 7, size=n_samples)
        
        # Generate square footage (600-3000 sq ft)
        square_footage = 500 + bedrooms * 300 + np.random.normal(0, 200, size=n_samples)
        square_footage = np.clip(square_footage, 600, 3000).astype(int)
        
        # Generate prices with some noise
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
    
    def train(self, X, y):
        """Train the linear regression model."""
        self.model.fit(X, y)
        self.is_trained = True
        
    def predict(self, X):
        """Make predictions using the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        return self.model.predict(X)
    
    def get_model_params(self):
        """Return the model parameters (coefficients and intercept)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting parameters!")
        
        return {
            'intercept': self.model.intercept_,
            'coefficients': {
                'bedrooms': self.model.coef_[0],
                'square_footage': self.model.coef_[1]
            }
        }
        
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation!")
            
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }