import logging
import json
from typing import Dict
import google.generativeai as genai
from google.protobuf.json_format import MessageToDict
from config.settings import (
    gemini_key_rotator,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GEMINI_TOP_P,
    GEMINI_TOP_K,
    GEMINI_MAX_OUTPUT_TOKENS,
)

def create_genai_model(max_retries=3):
    """
    Create a Generative AI model with key rotation and retry logic.

    :param max_retries: Maximum number of key rotation attempts
    :return: Configured GenerativeModel instance
    """
    logger = logging.getLogger(__name__)

    # Safety settings to reduce blocking
    safety_settings = {
        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
    }

    generation_config = {
        "temperature": GEMINI_TEMPERATURE,
        "top_p": GEMINI_TOP_P,
        "top_k": GEMINI_TOP_K,
        "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
    }

    # Track used keys to prevent repeated failures
    used_keys = set()

    for attempt in range(max_retries):
        try:
            # Get a new key
            current_key = gemini_key_rotator.get_key()

            # Skip keys we've already tried in this round
            if current_key in used_keys:
                continue

            # Configure Gemini with the current key
            genai.configure(api_key=current_key)

            # Create model
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

            return model

        except Exception as e:
            logger.warning(f"Key rotation attempt {attempt + 1} failed: {e}")

            # Mark the current key as failed
            gemini_key_rotator.mark_key_failed(current_key)
            used_keys.add(current_key)

            # If we've exhausted all keys, raise the last exception
            if attempt == max_retries - 1 or gemini_key_rotator.get_key_count() <= len(
                used_keys
            ):
                logger.error("All Gemini API keys have failed.")
                raise

    raise RuntimeError("Unable to create Generative AI model after multiple attempts")

def call_gemini_function(
    function_name: str, function_schema: Dict, prompt: str, max_retries=3
) -> Dict:
    """
    Call Gemini with function calling capabilities.

    Args:
        function_name: Name of the function to call
        function_schema: JSON schema for the function parameters
        prompt: The prompt to send to Gemini
        max_retries: Maximum number of retries if function calling fails

    Returns:
        The function parameters returned by Gemini (JSON serializable)
    """
    logger = logging.getLogger(__name__)
    model = create_genai_model()

    # Define the function for Gemini
    function_declarations = [
        {
            "name": function_name,
            "description": function_schema.get(
                "description", f"Call function {function_name}"
            ),
            "parameters": function_schema,
        }
    ]

    # Create the request
    for attempt in range(max_retries):
        try:
            logger.info(f"Calling Gemini function {function_name}, attempt {attempt + 1}")

            response = model.generate_content(
                prompt,
                tools=[{"function_declarations": function_declarations}],
                tool_config={"function_calling_config": {"mode": "any"}},
            )

            # Check if function was called
            function_call = None
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if (
                        hasattr(part, "function_call")
                        and part.function_call
                    ):  # Changed to hasattr
                        function_call = MessageToDict(part.function_call._pb)
                        break
                if function_call:
                    break

            if not function_call:
                logger.warning(
                    f"Gemini did not call the function on attempt {attempt + 1}"
                )
                continue
            
            print(function_call)

            # Parse the function response
            if function_call.get("name") == function_name:
                return function_call.get("args", {})
        except Exception as e:
            logger.warning(f"Function calling attempt {attempt + 1} failed: {e}")

            # If we've exhausted all retries, raise the last exception
            if attempt == max_retries - 1:
                logger.error(
                    f"All function calling attempts for {function_name} have failed."
                )
                raise

    # If we get here, all attempts failed
    raise RuntimeError(
        f"Unable to call Gemini function {function_name} after {max_retries} attempts"
    )
