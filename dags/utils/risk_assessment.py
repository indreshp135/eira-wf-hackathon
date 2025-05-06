import os
import json
import logging
from datetime import datetime
from dags.utils.gemini_util import call_gemini_function

from config.settings import (
    RESULTS_FOLDER
)

# Import the transaction folder utilities
from dags.utils.transaction_folder import (
    get_transaction_folder, save_transaction_data, load_transaction_data
)

# Configure logging
logger = logging.getLogger(__name__)

# Define the risk assessment function schema
RISK_ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "transaction_id": {"type": "string"},
        "extracted_entities": {
            "type": "array",
            "items": {"type": "string"}
        },
        "entity_types": {
            "type": "array",
            "items": {"type": "string"}
        },
        "risk_score": {
            "type": "number",
            "description": "Risk score between 0 and 1 (0 = low risk, 1 = high risk)"
        },
        "supporting_evidence": {
            "type": "array",
            "items": {"type": "string"}
        },
        "confidence_score": {
            "type": "number",
            "description": "Confidence score between 0 and 1"
        },
        "reason": {
            "type": "string",
            "description": "Detailed explanation of the risk assessment"
        },
        "timestamp": {
            "type": "string",
            "format": "date-time"
        }
    },
    "required": ["extracted_entities", "entity_types", "risk_score", "supporting_evidence", "confidence_score", "reason"]
}

def generate_risk_assessment(transaction_data=None, transaction_id=None, transaction_filepath=None, all_results=None, **context):
    """
    Generate a final risk assessment based on all collected data using Gemini function calling.
    
    Args:
        transaction_data: Text of the transaction (preferred)
        transaction_id: ID of the transaction
        all_results: Compiled results from all previous tasks
        context: Airflow task context
    """
    try:
        # Handle the transaction text - either directly provided or read from file
        if transaction_data:
            # New mode - use the provided text directly
            transaction_text = transaction_data
        else:
            raise ValueError("Either transaction_data or transaction_filepath must be provided")
        
        # Ensure we have a transaction ID
        if not transaction_id:
            transaction_id = f"txn_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            logger.warning(f"No transaction ID provided, using generated ID: {transaction_id}")
        
        # Get the transaction folder
        transaction_folder = get_transaction_folder(RESULTS_FOLDER, transaction_id)
        
        # Get the extracted entities from all_results or load from transaction folder
        if all_results and 'entities' in all_results:
            entities = all_results['entities']
        else:
            # Try to get entities from XCom if all_results wasn't passed
            ti = context.get('ti')
            if ti:
                entities = ti.xcom_pull(task_ids='extract_entities')
            else:
                entities = {}
            
            # If still not found, try to load from the transaction folder
            if not entities:
                entities = load_transaction_data(RESULTS_FOLDER, transaction_id, "entities.json")
                
        # Format the assessment_data structure
        assessment_data = {
            "transaction_text": transaction_text,
            "transaction_id": transaction_id,
            "extracted_entities": entities,
            "organizations": {},
            "people": {},
            "wikidata_people": {}
        }
        
        # Add organization results
        if all_results and 'organizations' in all_results:
            assessment_data["organizations"] = all_results['organizations']
        else:
            # Get results from organization folders
            org_results_path = os.path.join(transaction_folder, "organization_results")
            if os.path.exists(org_results_path):
                for subfolder in ['opencorporates', 'sanctions', 'wikidata', 'news']:
                    subfolder_path = os.path.join(org_results_path, subfolder)
                    if os.path.exists(subfolder_path):
                        for filename in os.listdir(subfolder_path):
                            if filename.endswith('.json'):
                                org_name = filename.replace('.json', '').replace('_', ' ')
                                if org_name not in assessment_data["organizations"]:
                                    assessment_data["organizations"][org_name] = {}
                                
                                with open(os.path.join(subfolder_path, filename), 'r', encoding='utf-8') as f:
                                    try:
                                        data = json.load(f)
                                        assessment_data["organizations"][org_name][subfolder] = data
                                    except Exception as e:
                                        logger.error(f"Error loading {subfolder} data for {org_name}: {str(e)}")
        
        # Add people results
        if all_results and 'people' in all_results:
            assessment_data["people"] = all_results['people']
        else:
            # Get results from people folders
            people_results_path = os.path.join(transaction_folder, "people_results")
            if os.path.exists(people_results_path):
                for subfolder in ['pep', 'sanctions', 'news']:
                    subfolder_path = os.path.join(people_results_path, subfolder)
                    if os.path.exists(subfolder_path):
                        for filename in os.listdir(subfolder_path):
                            if filename.endswith('.json'):
                                person_name = filename.replace('.json', '').replace('_', ' ')
                                if person_name not in assessment_data["people"]:
                                    assessment_data["people"][person_name] = {}
                                
                                with open(os.path.join(subfolder_path, filename), 'r', encoding='utf-8') as f:
                                    try:
                                        data = json.load(f)
                                        assessment_data["people"][person_name][subfolder] = data
                                    except Exception as e:
                                        logger.error(f"Error loading {subfolder} data for {person_name}: {str(e)}")
        
        # Add wikidata people results
        discovered_people_file = os.path.join(transaction_folder, "wikidata_discovered_people.json")
        if os.path.exists(discovered_people_file):
            try:
                with open(discovered_people_file, 'r', encoding='utf-8') as f:
                    assessment_data["wikidata_people"] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading discovered people data: {str(e)}")
        elif all_results and 'discovered_people' in all_results:
            assessment_data["wikidata_people"] = all_results['discovered_people']
        
        # Save the raw data for debugging and auditing
        save_transaction_data(RESULTS_FOLDER, transaction_id, "raw_assessment_data.json", assessment_data)
        logger.info(f"Saved raw assessment data to transaction folder")
        
        # Create a prompt for risk assessment
        prompt = f"""
        You are a financial crime expert specialized in Anti-Money Laundering (AML) risk assessment.
        
        Based on the following transaction data and associated information, generate a comprehensive risk assessment:
        
        TRANSACTION:
        {transaction_text}
        
        EXTRACTED ENTITIES AND VERIFICATION RESULTS:
        {json.dumps(assessment_data, indent=2)}
        
        Your task is to:
        1. Analyze the data and identify risk factors
        2. Determine if any parties are on sanctions lists
        3. Check if any individuals are Politically Exposed Persons (PEPs)
        4. Evaluate adverse news and negative publicity
        5. Assess jurisdictional risks
        6. Calculate an overall risk score between 0 and 1 (0 = low risk, 1 = high risk)
        
        For any data that couldn't be fetched successfully, acknowledge that but still make your best assessment with the available information.
        """
        
        # Call Gemini with function calling
        risk_assessment = call_gemini_function(
            function_name="generate_risk_assessment", 
            function_schema=RISK_ASSESSMENT_SCHEMA, 
            prompt=prompt
        )
        
        # Ensure we have the transaction ID and timestamp
        if "transaction_id" not in risk_assessment or not risk_assessment["transaction_id"]:
            risk_assessment["transaction_id"] = transaction_id
            
        if "timestamp" not in risk_assessment or not risk_assessment["timestamp"]:
            risk_assessment["timestamp"] = datetime.now().isoformat()
        
        # Save the risk assessment to the transaction folder
        save_transaction_data(RESULTS_FOLDER, transaction_id, "risk_assessment.json", risk_assessment)
        logger.info(f"Saved risk assessment to transaction folder")
        
        return risk_assessment
        
    except Exception as e:
        logger.error(f"Error generating risk assessment: {str(e)}")
        # Return a basic error response so the pipeline doesn't fail completely
        error_result = {
            "transaction_id": transaction_id or "unknown",
            "error": str(e),
            "status": "failed",
            "extracted_entities": [],
            "entity_types": [],
            "risk_score": 0.5,  # Default to medium risk when we can't assess
            "supporting_evidence": ["Error during risk assessment"],
            "confidence_score": 0.0,
            "reason": f"Could not complete risk assessment due to error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        
        # Save the error result to the transaction folder
        if transaction_id:
            save_transaction_data(RESULTS_FOLDER, transaction_id, "error.json", error_result)
            
        return error_result