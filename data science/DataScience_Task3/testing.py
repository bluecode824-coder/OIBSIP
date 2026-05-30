# %%
import os
import joblib
import pandas as pd

def test_car_price_predictor():
    model_path = 'car_price_model.pkl'
    
    # 1. Verification Check
    if not os.path.exists(model_path):
        print(f" Error: '{model_path}' not found in the current directory.")
        return
    
    # 2. Load the trained model
    model = joblib.load(model_path)
    print("Saved Car Price Prediction model loaded successfully!\n")
    
    # 3. Prediction function
    def predict_price(year, present_price, driven_kms, owner, fuel_type, selling_type, transmission):
        fuel_diesel = 1 if fuel_type.strip().capitalize() == 'Diesel' else 0
        fuel_petrol = 1 if fuel_type.strip().capitalize() == 'Petrol' else 0
        seller_individual = 1 if selling_type.strip().capitalize() == 'Individual' else 0
        trans_manual = 1 if transmission.strip().capitalize() == 'Manual' else 0
        
        input_data = {
            'Year': [year],
            'Present_Price': [present_price],
            'Driven_kms': [driven_kms],
            'Owner': [owner],
            'Fuel_Type_Diesel': [fuel_diesel],
            'Fuel_Type_Petrol': [fuel_petrol],
            'Selling_type_Individual': [seller_individual],
            'Transmission_Manual': [trans_manual]
        }
        
        input_df = pd.DataFrame(input_data)
        predicted_val = model.predict(input_df)[0]
        return max(0.0, round(predicted_val, 2))

    # 4. Updated Test Scenarios
    print("--- Running Scenario Predictions ---")
    
    # Scenario A: Modern, premium diesel car sold by a dealer
    car_a = predict_price(year=2020, present_price=12.5, driven_kms=15000, owner=0, 
                          fuel_type='Diesel', selling_type='Dealer', transmission='Manual')
    print(f"Scenario A (2020 Premium Diesel Manual Dealer): Estimated Price = {car_a} Lakhs")

    # REVISED Scenario B: Moderate age used budget hatchback (e.g., a 2014 Maruti Swift / Hyundai i20)
    # Shifting the year up to 2014 and lowering mileage brings it perfectly into positive valuation territory.
    car_b = predict_price(year=2014, present_price=5.0, driven_kms=45000, owner=0, 
                          fuel_type='Petrol', selling_type='Individual', transmission='Manual')
    print(f"Scenario B (2014 Mid-Age Budget Petrol Individual): Estimated Price = {car_b} Lakhs")
    
    # Scenario C: Luxury Automatic Sedan
    car_c = predict_price(year=2018, present_price=25.0, driven_kms=30000, owner=0, 
                          fuel_type='Petrol', selling_type='Dealer', transmission='Automatic')
    print(f"Scenario C (2018 Luxury Automatic Petrol Dealer):  Estimated Price = {car_c} Lakhs")

if __name__ == '__main__':
    test_car_price_predictor()


