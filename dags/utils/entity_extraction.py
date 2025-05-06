import os
import json
import logging
from dags.utils.gemini_util import call_gemini_function

from dags.config.settings import (
    RESULTS_FOLDER
)

# Import the transaction folder utilities
from dags.utils.transaction_folder import (
    get_transaction_folder, save_transaction_data, load_transaction_data
)

# Configure logging
logger = logging.getLogger(__name__)

# Define the entity extraction function schema
ENTITY_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "transaction_id": {"type": "string"},
        "organizations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string", "enum": ["sender", "recipient", "intermediary"]},
                    "jurisdiction": {"type": "string"},
                    "entity_type": {"type": "string", "enum": ["corporation", "shell_company", "non_profit", "government_agency", "financial_institution"]}
                },
                "required": ["name"]
            }
        },
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string", "enum": ["director", "approver", "beneficiary", "other"]},
                    "country": {"type": "string"}
                },
                "required": ["name"]
            }
        },
        "transaction": {
            "type": "object",
            "properties": {
                "amount": {"type": "string"},
                "currency": {"type": "string"},
                "purpose": {"type": "string"},
                "date": {"type": "string"}
            }
        },
        "jurisdictions": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["transaction_id", "organizations", "people"]
}


def extract_entities_from_text(transaction_text, transaction_id, **context):
    """
    Use Gemini with function calling to extract entities from transaction text.
    
    Args:
        transaction_text: The raw text of the transaction
        transaction_id: The ID of the transaction
        context: Airflow task context
        
    Returns:
        Dictionary of extracted entities
    """
    try:
        logger.info(f"Extracting entities for transaction: {transaction_id}")
        
        if not transaction_text:
            raise ValueError("Transaction text is empty")
        
        # Create the transaction folder
        transaction_folder = get_transaction_folder(RESULTS_FOLDER, transaction_id)
        
        # Save the original transaction text
        with open(os.path.join(transaction_folder, "transaction.txt"), "w", encoding="utf-8") as f:
            f.write(transaction_text)
        
        # Create a prompt for entity extraction
        prompt = f"""
        You are a financial crime expert. Extract entities from the following transaction data:

        {transaction_text}

        Please identify:
        1. Organizations involved (sender and recipient companies/entities)
        2. People mentioned (directors, approvers, beneficiaries)
        3. Transaction details (amount, currency, purpose)
        4. Jurisdictions mentioned (countries, territories)

        Provide a structured response with all the entities you identified from the transaction data.
        """

        # Call Gemini with function calling
        entities = call_gemini_function(
            function_name="extract_entities",
            function_schema=ENTITY_EXTRACTION_SCHEMA,
            prompt=prompt
        )
        
        # Ensure transaction_id is set correctly
        entities["transaction_id"] = transaction_id
        
        # Log the result
        logger.info(f"Extracted entities: {entities}")
        
        # Save the extracted entities to the transaction folder
        save_transaction_data(RESULTS_FOLDER, transaction_id, "entities.json", entities)
            
        return entities
    
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        raise