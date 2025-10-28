# src/pipeline/location_service.py
import json
import os
from src.exceptions import CustomException
from src.logger import logging

class LocationService:
    def __init__(self):
                self.locations_file = os.path.join(os.getcwd(), 'artifacts', 'locations.json')
                # print(f"Locations file path set to: {os.getcwd()}")

    
    def get_all_locations(self):
        """
        Get all available locations from JSON file
        """
        try:
            if os.path.exists(self.locations_file):
                with open(self.locations_file, 'r') as f:
                    locations_data = json.load(f)
                locations = locations_data.get('locations', [])
                logging.info(f"Loaded {len(locations)} locations from {self.locations_file}")
                return locations
            else:
                logging.warning(f"Locations file not found: {self.locations_file}")
                return []
            
        except Exception as e:
            logging.error(f"Error loading locations: {e}")
            return []
    
    def get_location_suggestions(self, query: str = ""):
        """
        Get location suggestions based on user input
        """
        try:
            all_locations = self.get_all_locations()
            query = query.lower().strip()
            
            if not query:
                return all_locations[:10]  # Return first 10 if no query
            
            # Filter locations that contain the query
            suggestions = [loc for loc in all_locations if query in loc.lower()]
            return suggestions[:10]  # Limit to 10 suggestions
            
        except Exception as e:
            logging.error(f"Error getting suggestions: {e}")
            return []
 
    def validate_location(self, location):
        """
        Check if a location exists in our data
        """
        try:
            all_locations = self.get_all_locations()
            # Convert both to lowercase for case-insensitive comparison
            location_lower = location.lower().strip()
            return any(loc.lower() == location_lower for loc in all_locations)
        except Exception as e:
            logging.error(f"Error validating location: {e}")
            return False

# Create a global instance for easy access
location_service = LocationService()