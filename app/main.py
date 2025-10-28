from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.location_service import location_service

# Initialize FastAPI app
app = FastAPI(
    title="Bengaluru House Price Prediction API",
    description="API for predicting house prices in Bengaluru based on property characteristics",
    version="1.0.0"
)

# CORS middleware - IMPORTANT for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic model for API validation
class HouseData(BaseModel):
    location: str
    sqft: float
    bath: float
    bhk: float

    # Example values for API documentation
    class Config:
        schema_extra = {
            "example": {
                "location": "Whitefield",
                "sqft": 1250.0,
                "bath": 2.0,
                "bhk": 2.0
            }
        }

# Main prediction endpoint
@app.post("/predict")
async def predict_price(house_data: HouseData):
    try:
        # Validate location exists (case-insensitive)
        if not location_service.validate_location(house_data.location):
            available_locations = location_service.get_all_locations()[:5]
            # Suggest case-correct version if similar exists
            user_input_lower = house_data.location.lower()
            similar_locations = [loc for loc in available_locations if user_input_lower in loc.lower()]
            
            if similar_locations:
                message = f"Location '{house_data.location}' not found. Did you mean: {', '.join(similar_locations)}?"
            else:
                message = f"Location '{house_data.location}' not found. Available locations include: {', '.join(available_locations)}"
            
            return {
                "status": "error",
                "message": message
            }
        # Create CustomData object from Pydantic model
        data = CustomData(
            location=house_data.location,
            sqft=house_data.sqft,
            bath=house_data.bath,
            bhk=house_data.bhk
        )

        # Get data as DataFrame
        pred_df = data.get_data_as_dataframe()
        
        # Make prediction
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(pred_df)

        return {
            "status": "success",
            "predicted_price": round(pred[0], 2),
            "formatted_price": f"₹{round(pred[0], 2):,}",
            "input_data": house_data.dict()
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Bengaluru House Price Prediction API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is operational"}

# Get all available locations
@app.get("/locations")
async def get_locations():
    """Get all available locations for dropdown/autocomplete"""
    locations = location_service.get_all_locations()
    return {
        "status": "success",
        "locations": locations,
        "count": len(locations)
    }

# Get location suggestions for autocomplete
@app.get("/locations/suggest")
async def get_location_suggestions(query: str = Query("", description="Search query for locations")):
    """Get location suggestions for autocomplete based on search query"""
    suggestions = location_service.get_location_suggestions(query)
    return {
        "status": "success",
        "suggestions": suggestions,
        "count": len(suggestions),
        "query": query
    }

# Get available options for input ranges
@app.get("/options")
async def get_options():
    """Get available options and recommended ranges for input fields"""
    return {
        "location_count": len(location_service.get_all_locations()),
        "recommended_ranges": {
            "sqft": {"min": 500, "max": 5000, "recommended": 1000},
            "bath": {"min": 1, "max": 10, "recommended": 2},
            "bhk": {"min": 1, "max": 10, "recommended": 2}
        },
        "popular_locations": location_service.get_all_locations()[:10]  # Top 10 locations
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)