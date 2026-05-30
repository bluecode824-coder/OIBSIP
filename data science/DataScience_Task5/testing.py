# %%
import joblib
import pandas as pd

# Load model back into memory 
loaded_sales_model = joblib.load('sales_prediction_rf_model.pkl')

def forecast_sales(tv_budget, radio_budget, newspaper_budget):
    # Match the exact feature naming conventions of training data
    input_data = pd.DataFrame(
        [[tv_budget, radio_budget, newspaper_budget]], 
        columns=['TV', 'Radio', 'Newspaper']
    )
    
    # Generate prediction
    prediction = loaded_sales_model.predict(input_data)[0]
    return round(prediction, 2)

# --- Business Budget Scenarios ---
# Scenario: $200k TV, $40k Radio, $20k Newspaper
predicted_revenue = forecast_sales(200.0, 40.0, 20.0)

print(f"Proposed Marketing Budgets:")
print(f"  - TV: $200,000 | Radio: $40,000 | Newspaper: $20,000")
print(f"Estimated Sales Forecast: {predicted_revenue} units/units worth")


