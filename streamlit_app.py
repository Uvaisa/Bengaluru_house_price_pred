import streamlit as st
import requests
import json

# Configure the page
st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        # margin-bottom: 1rem;
        # margin-top: 0.5rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .price-result {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ff4b4b;
        text-align: center;
    }
    .info-section {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .location-suggestion {
        background-color: #e1f5fe;
        padding: 0.5rem;
        margin: 0.2rem;
        border-radius: 5px;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# API configuration
API_URL = "http://localhost:8000"  # Change if deployed

def predict_price(data):
    """Send prediction request to FastAPI backend"""
    try:
        response = requests.post(f"{API_URL}/predict", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": f"API Error: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Connection error: {str(e)}"}

def get_location_suggestions(query):
    """Get location suggestions from API"""
    try:
        response = requests.get(f"{API_URL}/locations/suggest", params={"query": query})
        if response.status_code == 200:
            return response.json().get("suggestions", [])
        return []
    except:
        return []

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Bengaluru House Price Predictor</h1>', unsafe_allow_html=True)
    
    # Create three columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📍 Location Details")
        
        # Get all locations for dropdown
        try:
            response = requests.get(f"{API_URL}/locations")
            all_locations = []
            if response.status_code == 200:
                all_locations = response.json().get("locations", [])
        except:
            all_locations = []
        
        # Location input with autocomplete
        location_input = st.text_input(
            "Enter Location", 
            placeholder="Start typing (e.g., Whitefield, Koramangala...)",
            help="Location names will autocomplete as you type"
        )
        
        # Show location suggestions
        if location_input:
            suggestions = get_location_suggestions(location_input)
            if suggestions:
                st.write("💡 Suggestions:")
                cols = st.columns(3)
                for i, suggestion in enumerate(suggestions[:6]):  # Show max 6 suggestions
                    with cols[i % 3]:
                        if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                            st.session_state.selected_location = suggestion
                
                # Update input if suggestion clicked
                if 'selected_location' in st.session_state:
                    location_input = st.session_state.selected_location
                    del st.session_state.selected_location
    
    with col2:
        st.subheader("📏 Property Specifications")
        sqft = st.slider("Total Square Feet", min_value=500, max_value=5000, value=1250, step=50)
        bath = st.slider("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
        bhk = st.slider("BHK", min_value=1, max_value=10, value=2, step=1)
    
    with col3:
        st.subheader("📊 Quick Stats")
        
        # Display property stats
        st.metric("Square Feet", sqft)
        st.metric("Bathrooms", bath)
        st.metric("BHK", bhk)
        
        # Show property type based on BHK
        property_type = "Studio" if bhk == 1 else f"{bhk} BHK"
        st.metric("Property Type", property_type)
    
    # Predict button
    if st.button("🚀 Predict Price", use_container_width=True, type="primary"):
        if not location_input:
            st.error("❌ Please enter a location")
            return
            
        # Prepare data for API
        input_data = {
            "location": location_input,
            "sqft": float(sqft),
            "bath": float(bath),
            "bhk": float(bhk)
        }

        
        # Show loading spinner
        with st.spinner("Calculating price..."):
            result = predict_price(input_data)
        
        # Display results
        if result["status"] == "success":
            st.markdown("---")
            st.subheader("💰 Predicted Price")
            predicted_price = result["predicted_price"]
            formatted_price = result["formatted_price"]
            
            st.markdown(f'<div class="price-result">{formatted_price}</div>', unsafe_allow_html=True)
            
            # Additional price insights
            st.success("✅ Price prediction completed successfully!")        
            st.markdown('</div>', unsafe_allow_html=True)
                
        else:
            st.error(f"❌ {result['message']}")
    
    # Footer with additional information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p><strong>Note:</strong> Predictions are based on historical data and machine learning models. 
    Actual market prices may vary based on additional factors like amenities, floor, age of property, etc.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    