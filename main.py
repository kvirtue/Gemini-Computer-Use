from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai
from google.genai import types
from google.genai.types import FunctionResponse, FinishReason
from playwright.sync_api import sync_playwright
import base64
import time
import os
import logging
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Screen dimensions for the browser
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900

# Screenshot memory optimization - only keep last N turns with screenshots
MAX_RECENT_TURNS_WITH_SCREENSHOTS = 3

# List of predefined Computer Use functions (for screenshot cleanup)
PREDEFINED_COMPUTER_USE_FUNCTIONS = [
    "open_web_browser",
    "click_at",
    "hover_at",
    "type_text_at",
    "scroll_document",
    "scroll_at",
    "wait_5_seconds",
    "go_back",
    "go_forward",
    "search",
    "navigate",
    "key_combination",
    "drag_and_drop",
]

# Playwright key mapping for key_combination
PLAYWRIGHT_KEY_MAP = {
    "backspace": "Backspace",
    "tab": "Tab",
    "return": "Enter",
    "enter": "Enter",
    "shift": "Shift",
    "control": "Control",
    "alt": "Alt",
    "escape": "Escape",
    "space": "Space",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "end": "End",
    "home": "Home",
    "left": "ArrowLeft",
    "up": "ArrowUp",
    "right": "ArrowRight",
    "down": "ArrowDown",
    "insert": "Insert",
    "delete": "Delete",
    "f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4",
    "f5": "F5", "f6": "F6", "f7": "F7", "f8": "F8",
    "f9": "F9", "f10": "F10", "f11": "F11", "f12": "F12",
    "command": "Meta",
    "meta": "Meta",
}

# Lazy client initialization
_client = None


def get_client():
    """Get or create the Gemini client (lazy initialization)."""
    global _client
    if _client is None:
        # Use Gemini API with API key (Computer Use model not on Vertex AI yet)
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key:
            _client = genai.Client(api_key=api_key)
        else:
            # Fallback: try to get from Secret Manager
            from google.cloud import secretmanager
            sm_client = secretmanager.SecretManagerServiceClient()
            project = os.environ.get('GOOGLE_CLOUD_PROJECT', 'gemini-browser-agent-480208')
            secret_name = f"projects/{project}/secrets/gemini-api-key/versions/latest"
            response = sm_client.access_secret_version(name=secret_name)
            api_key = response.payload.data.decode('UTF-8')
            _client = genai.Client(api_key=api_key)
    return _client


def denormalize_x(x: int) -> int:
    """Convert normalized x coordinate (0-999) to actual pixel coordinate."""
    return int(x / 1000 * SCREEN_WIDTH)


def denormalize_y(y: int) -> int:
    """Convert normalized y coordinate (0-999) to actual pixel coordinate."""
    return int(y / 1000 * SCREEN_HEIGHT)


def get_model_response(client, model_name, contents, config, max_retries=5, base_delay_s=1):
    """Get model response with retry logic and exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            return response
        except Exception as e:
            logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                delay = base_delay_s * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"API call failed after {max_retries} attempts")
                raise


def cleanup_old_screenshots(contents):
    """Remove screenshots from older turns to reduce token usage."""
    turns_with_screenshots = 0
    
    for content in reversed(contents):
        if content.role == "user" and content.parts:
            has_screenshot = False
            for part in content.parts:
                if (part.function_response 
                    and part.function_response.parts 
                    and part.function_response.name in PREDEFINED_COMPUTER_USE_FUNCTIONS):
                    has_screenshot = True
                    break
            
            if has_screenshot:
                turns_with_screenshots += 1
                if turns_with_screenshots > MAX_RECENT_TURNS_WITH_SCREENSHOTS:
                    for part in content.parts:
                        if (part.function_response 
                            and part.function_response.parts 
                            and part.function_response.name in PREDEFINED_COMPUTER_USE_FUNCTIONS):
                            part.function_response.parts = None


def execute_key_combination(page, keys_str):
    """Execute a key combination like 'Control+A' or 'Enter'."""
    keys = keys_str.split("+")
    # Normalize keys to Playwright format
    keys = [PLAYWRIGHT_KEY_MAP.get(k.lower(), k) for k in keys]
    
    # Press all modifier keys
    for key in keys[:-1]:
        page.keyboard.down(key)
    
    # Press the final key
    page.keyboard.press(keys[-1])
    
    # Release modifier keys in reverse order
    for key in reversed(keys[:-1]):
        page.keyboard.up(key)


def execute_browser_task(task_description: str) -> dict:
    """
    Execute a browser task using Gemini Computer Use.

    Args:
        task_description: Natural language description of what to do
                         Example: "Go to Google and search for salesforce"

    Returns:
        dict with status, final_url, actions_taken, and final_response
    """
    # Start Playwright browser with security args
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(
        headless=True,
        args=[
            "--disable-extensions",
            "--disable-file-system",
            "--disable-plugins",
            "--disable-dev-shm-usage",
            "--disable-background-networking",
            "--disable-default-apps",
            "--disable-sync",
        ]
    )
    context = browser.new_context(
        viewport={"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT}
    )
    page = context.new_page()
    
    # Handle new tabs - redirect to same page (model only supports single tab)
    def handle_new_page(new_page):
        new_url = new_page.url
        new_page.close()
        page.goto(new_url)
    
    context.on("page", handle_new_page)

    try:
        # Navigate to starting page
        page.goto("https://www.google.com")
        page.wait_for_load_state()

        # Configure Gemini Computer Use with recommended settings
        config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            tools=[types.Tool(
                computer_use=types.ComputerUse(
                    environment=types.Environment.ENVIRONMENT_BROWSER
                )
            )],
        )

        # Take initial screenshot
        time.sleep(0.5)  # Allow page to fully render
        screenshot = page.screenshot(type="png")

        # Build initial conversation with Gemini
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(text=task_description),
                    types.Part.from_bytes(data=screenshot, mime_type='image/png')
                ]
            )
        ]

        # Initialize result tracking
        result = {
            "status": "in_progress",
            "actions_taken": [],
            "final_url": "",
            "final_response": ""
        }

        # Agent loop - max 15 turns (ROI tasks can take longer)
        for turn in range(15):
            logger.info(f"Turn {turn + 1}")

            # Ask Gemini what to do next (with retry logic)
            try:
                response = get_model_response(
                    client=get_client(),
                    model_name='gemini-2.5-computer-use-preview-10-2025',
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                result["status"] = "error"
                result["final_response"] = f"API error: {str(e)}"
                break

            if not response.candidates:
                logger.warning("Response has no candidates")
                result["status"] = "error"
                result["final_response"] = "Empty response from model"
                break

            candidate = response.candidates[0]
            
            # Append model response to conversation
            if candidate.content:
                contents.append(candidate.content)

            # Check for malformed function call - retry
            if (candidate.finish_reason == FinishReason.MALFORMED_FUNCTION_CALL 
                and not any(part.function_call for part in candidate.content.parts if candidate.content and candidate.content.parts)):
                logger.warning("Malformed function call, retrying...")
                continue

            # Check if Gemini is done (no more function calls)
            has_function_calls = False
            if candidate.content and candidate.content.parts:
                has_function_calls = any(
                    part.function_call for part in candidate.content.parts
                )

            if not has_function_calls:
                # Extract final text response
                text_parts = []
                if candidate.content and candidate.content.parts:
                    text_parts = [part.text for part in candidate.content.parts if part.text]
                text_response = " ".join(text_parts)
                result["status"] = "completed"
                result["final_response"] = text_response
                logger.info(f"Task completed: {text_response[:100]}...")
                break

            # Execute each function call from Gemini
            function_responses = []

            if not candidate.content or not candidate.content.parts:
                logger.warning("Candidate has no content or parts")
                continue

            for part in candidate.content.parts:
                if part.function_call:
                    fc = part.function_call
                    action_name = fc.name
                    args = fc.args or {}

                    logger.info(f"Executing action: {action_name}")

                    # Log the action
                    result["actions_taken"].append({
                        "action": action_name,
                        "args": dict(args) if args else {}
                    })

                    # Check for safety decision
                    extra_fr_fields = {}
                    if args.get("safety_decision"):
                        safety = args["safety_decision"]
                        logger.warning(f"Safety decision required: {safety.get('explanation', 'No explanation')}")
                        # In serverless, we auto-acknowledge (log for audit)
                        extra_fr_fields["safety_acknowledgement"] = "true"

                    try:
                        # Execute the action based on type
                        if action_name == "open_web_browser":
                            # Browser is already open
                            pass

                        elif action_name == "click_at":
                            x = denormalize_x(args["x"])
                            y = denormalize_y(args["y"])
                            page.mouse.click(x, y)

                        elif action_name == "hover_at":
                            x = denormalize_x(args["x"])
                            y = denormalize_y(args["y"])
                            page.mouse.move(x, y)

                        elif action_name == "type_text_at":
                            x = denormalize_x(args["x"])
                            y = denormalize_y(args["y"])
                            page.mouse.click(x, y)
                            page.wait_for_load_state()
                            
                            # Clear existing text (Linux-compatible)
                            clear_before = args.get("clear_before_typing", True)
                            if clear_before:
                                execute_key_combination(page, "Control+A")
                                execute_key_combination(page, "Delete")
                            
                            # Type new text
                            page.keyboard.type(args["text"])
                            page.wait_for_load_state()
                            
                            if args.get("press_enter", False):
                                execute_key_combination(page, "Enter")

                        elif action_name == "scroll_document":
                            direction = args["direction"]
                            if direction == "down":
                                execute_key_combination(page, "PageDown")
                            elif direction == "up":
                                execute_key_combination(page, "PageUp")
                            elif direction == "left":
                                scroll_amount = SCREEN_WIDTH // 2
                                page.evaluate(f"window.scrollBy(-{scroll_amount}, 0)")
                            elif direction == "right":
                                scroll_amount = SCREEN_WIDTH // 2
                                page.evaluate(f"window.scrollBy({scroll_amount}, 0)")

                        elif action_name == "scroll_at":
                            x = denormalize_x(args["x"])
                            y = denormalize_y(args["y"])
                            direction = args["direction"]
                            magnitude = args.get("magnitude", 800)
                            
                            # Denormalize magnitude based on direction
                            if direction in ("up", "down"):
                                magnitude = denormalize_y(magnitude)
                            else:
                                magnitude = denormalize_x(magnitude)
                            
                            page.mouse.move(x, y)
                            
                            dx, dy = 0, 0
                            if direction == "up":
                                dy = -magnitude
                            elif direction == "down":
                                dy = magnitude
                            elif direction == "left":
                                dx = -magnitude
                            elif direction == "right":
                                dx = magnitude
                            
                            page.mouse.wheel(dx, dy)

                        elif action_name == "navigate":
                            url = args["url"]
                            # Normalize URL
                            if not url.startswith(("http://", "https://")):
                                url = "https://" + url
                            page.goto(url)

                        elif action_name == "search":
                            page.goto("https://www.google.com")

                        elif action_name == "go_back":
                            page.go_back()

                        elif action_name == "go_forward":
                            page.go_forward()

                        elif action_name == "wait_5_seconds":
                            time.sleep(5)

                        elif action_name == "key_combination":
                            keys = args["keys"]
                            execute_key_combination(page, keys)

                        elif action_name == "drag_and_drop":
                            x = denormalize_x(args["x"])
                            y = denormalize_y(args["y"])
                            dest_x = denormalize_x(args["destination_x"])
                            dest_y = denormalize_y(args["destination_y"])
                            
                            page.mouse.move(x, y)
                            page.wait_for_load_state()
                            page.mouse.down()
                            page.wait_for_load_state()
                            page.mouse.move(dest_x, dest_y)
                            page.wait_for_load_state()
                            page.mouse.up()

                        else:
                            logger.warning(f"Unknown action: {action_name}")

                        # Wait for page to settle after action
                        page.wait_for_load_state()
                        time.sleep(0.5)

                    except Exception as e:
                        logger.error(f"Error executing {action_name}: {e}")

                    # Capture new screenshot after action
                    new_screenshot = page.screenshot(type="png")

                    # Build function response for Gemini
                    function_responses.append(
                        FunctionResponse(
                            name=action_name,
                            response={"url": page.url, **extra_fr_fields},
                            parts=[types.FunctionResponsePart(
                                inline_data=types.FunctionResponseBlob(
                                    mime_type="image/png",
                                    data=new_screenshot
                                )
                            )]
                        )
                    )

            # Add function responses to conversation
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(function_response=fr) for fr in function_responses]
                )
            )

            # Clean up old screenshots to reduce token usage
            cleanup_old_screenshots(contents)

        # Capture final state
        final_screenshot = page.screenshot(type="png")
        result["final_screenshot_b64"] = base64.b64encode(final_screenshot).decode()
        result["final_url"] = page.url

        return result

    finally:
        browser.close()
        playwright.stop()


def get_lucidchart_credentials():
    """
    Get Lucidchart login credentials from environment variables or Secret Manager.
    
    Returns:
        tuple: (email, password)
    
    Raises:
        ValueError: If credentials cannot be found
    """
    email = os.environ.get('LUCIDCHART_EMAIL')
    password = os.environ.get('LUCIDCHART_PASSWORD')
    
    if email and password:
        return (email, password)
    
    # Fallback: try to get from Secret Manager
    try:
        from google.cloud import secretmanager
        sm_client = secretmanager.SecretManagerServiceClient()
        project = os.environ.get('GOOGLE_CLOUD_PROJECT', 'gemini-browser-agent-480208')
        
        email_secret_name = f"projects/{project}/secrets/lucidchart-email/versions/latest"
        password_secret_name = f"projects/{project}/secrets/lucidchart-password/versions/latest"
        
        email_response = sm_client.access_secret_version(name=email_secret_name)
        password_response = sm_client.access_secret_version(name=password_secret_name)
        
        email = email_response.payload.data.decode('UTF-8')
        password = password_response.payload.data.decode('UTF-8')
        
        return (email, password)
    except Exception as e:
        logger.error(f"Failed to get Lucidchart credentials: {e}")
        raise ValueError("Lucidchart credentials not found in environment variables or Secret Manager")


def build_diagram_task_instructions(data):
    """
    Build detailed task instructions for creating a Lucidchart architecture diagram.
    
    Args:
        data: dict with opportunity_id, company_name, industry, products
    
    Returns:
        str: Detailed task instructions for Gemini
    """
    company_name = data.get('company_name', 'Company')
    industry = data.get('industry', '')
    products = data.get('products', [])
    
    # Get credentials
    try:
        email, password = get_lucidchart_credentials()
    except ValueError as e:
        raise ValueError(f"Cannot build diagram task: {e}")
    
    # Build products list
    products_text = ", ".join(products) if products else "no products"
    
    instructions = f"""Create a Lucidchart architecture diagram for {company_name}.

Step-by-step instructions:

1. Navigate to https://lucid.co and wait for the page to load.

2. Find and click the "Sign In" or "Log In" button. Enter the following credentials:
   - Email: {email}
   - Password: {password}
   Click the login/submit button to sign in.

3. After logging in, create a new blank diagram:
   - Look for a "Create" button or "+ New" button
   - Select "Blank Diagram" or "New Document"
   - Wait for the diagram editor to load

4. For each of the following products, add their official logo/icon to the canvas:
   {products_text}
   
   For each product:
   - Use the shape library search (usually in a sidebar or toolbar)
   - Search for the product name (e.g., "Salesforce", "MuleSoft", "Tableau")
   - Find and select the official logo/icon for that product
   - Drag it onto the canvas
   - Position it in a logical architecture layout (arrange components horizontally or in a flow)

5. Add connector arrows between the components to show data flow and integrations:
   - Use the connector/arrow tool from the toolbar
   - Draw arrows connecting related components
   - Make sure the connections show logical data flow

6. Add labels to the connectors or near components to describe integrations:
   - Add labels like "Real-time CDC", "Einstein Analytics Feed", "API Integration", etc.
   - Use text boxes or label tools to add these descriptions
   - Position labels near the relevant connectors or components

7. Set the diagram title to "{company_name} Architecture":
   - Find the title field (usually at the top of the document)
   - Replace any existing title with "{company_name} Architecture"

8. Get the shareable link:
   - Look for a "Share" button or icon
   - Click it and ensure the document is shareable
   - Copy the shareable link URL
   - The URL should look like: https://lucid.app/documents/view/...

9. In your final response, report the shareable link URL clearly. Format it as: "Lucidchart URL: [the full URL]"

Make sure the diagram is well-organized and professional-looking."""
    
    return instructions


def build_roi_task_instructions(data):
    """
    Build detailed task instructions for populating Google Sheets ROI calculator.
    
    Args:
        data: dict with opportunity_id, company_name, total_initial_investment_cost,
              average_annual_cash_flow, annual_profit, roi_sheet_url
    
    Returns:
        str: Detailed task instructions for Gemini
    """
    company_name = data.get('company_name', 'Company')
    total_initial_investment_cost = data.get('total_initial_investment_cost', 0)
    average_annual_cash_flow = data.get('average_annual_cash_flow', 0)
    annual_profit = data.get('annual_profit', '')
    roi_sheet_url = data.get('roi_sheet_url', '')
    
    if not roi_sheet_url:
        raise ValueError("roi_sheet_url is required")
    
    instructions = f"""Populate a Google Sheets ROI calculator and extract the calculated values.

Step-by-step instructions:

1. Navigate to the Google Sheets URL: {roi_sheet_url}
   Wait for the sheet to fully load. The Inputs tab is already open (it's the first tab).

2. Locate and fill in the following cells in the Inputs tab:
   - Find the cell for "Company Name" (or similar label) and type: {company_name}
   - Find the cell for "Total Initial Investment Cost" (or similar label like "Initial Investment", "Investment Cost", "Total Investment") and type: {total_initial_investment_cost}
   - Find the cell for "Average Annual Cash Flow/Savings" (or similar label like "Annual Cash Flow", "Average Annual Savings", "Annual Savings") and type: {average_annual_cash_flow}
   - Find the cell for "Customer Annual Profit" or "Annual Profit" (or similar label) and type: {annual_profit}
   
   Make sure to click on each cell before typing, and press Enter or Tab after entering each value.

3. Navigate to the "Calculations" tab (or "Results" or "Output" tab):
   - Click on the tab at the bottom of the sheet
   - Wait for the formulas to recalculate (this may take a few seconds)

4. Wait for formulas to recalculate:
   - After switching to the Calculations tab, wait at least 5 seconds
   - Scroll through the sheet to ensure all calculations have updated

5. Read the calculated values from the Calculations tab:
   - Find the cell containing the "5-Year ROI" value (usually shown as a percentage like 412%)
   - Find the cell containing the "Payback Period" value (usually shown in months like 8)
   - Find the cell containing the "Annual Savings" value (usually shown as a dollar amount like $847,000)

6. Extract the numeric values:
   - For ROI: extract just the number (e.g., if it says "412%", extract 412)
   - For Payback Period: extract just the number of months (e.g., if it says "8 months", extract 8)
   - For Annual Savings: extract just the dollar amount as a number (e.g., if it says "$847,000", extract 847000)

7. In your final response, return the values in this EXACT JSON format (no other text, just the JSON):
{{
  "roi_percentage": <number>,
  "payback_months": <number>,
  "annual_savings": <number>
}}

For example, if ROI is 412%, Payback Period is 8 months, and Annual Savings is $847,000, return:
{{
  "roi_percentage": 412,
  "payback_months": 8,
  "annual_savings": 847000
}}

IMPORTANT: Your final response must contain ONLY valid JSON in this format. Do not include any explanatory text before or after the JSON."""
    
    return instructions


def extract_structured_data_from_response(final_response):
    """
    Extract structured ROI data from Gemini's final response.
    
    Args:
        final_response: str - Gemini's final text response
    
    Returns:
        dict with roi_percentage, payback_months, annual_savings
    
    Raises:
        ValueError: If data cannot be extracted
    """
    if not final_response:
        raise ValueError("Empty response from Gemini")
    
    # Try to find JSON in the response
    # Look for JSON object pattern
    json_match = re.search(r'\{[^{}]*"roi_percentage"[^{}]*\}', final_response, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # Validate required fields and handle None values
            if 'roi_percentage' in data and 'payback_months' in data and 'annual_savings' in data:
                roi_pct = data.get('roi_percentage')
                payback = data.get('payback_months')
                savings = data.get('annual_savings')
                
                # Check for None values
                if roi_pct is None or payback is None or savings is None:
                    raise ValueError("ROI values cannot be None")
                
                # Convert to int, handling string numbers
                try:
                    return {
                        'roi_percentage': int(float(roi_pct)) if roi_pct else 0,
                        'payback_months': int(float(payback)) if payback else 0,
                        'annual_savings': int(float(savings)) if savings else 0
                    }
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Could not convert ROI values to numbers: {e}")
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse JSON or convert values: {e}")
            pass
    
    # Fallback: try to extract values using regex patterns
    roi_match = re.search(r'roi[_\s]*percentage[:\s]*(\d+)', final_response, re.IGNORECASE)
    payback_match = re.search(r'payback[_\s]*period[:\s]*(\d+)', final_response, re.IGNORECASE)
    savings_match = re.search(r'annual[_\s]*savings[:\s]*\$?([\d,]+)', final_response, re.IGNORECASE)
    
    result = {}
    if roi_match:
        result['roi_percentage'] = int(roi_match.group(1))
    if payback_match:
        result['payback_months'] = int(payback_match.group(1))
    if savings_match:
        # Remove commas from number
        savings_str = savings_match.group(1).replace(',', '')
        result['annual_savings'] = int(savings_str)
    
    if len(result) == 3:
        return result
    
    # If we still don't have all values, raise an error
    raise ValueError(f"Could not extract ROI values from response: {final_response[:200]}...")


# Create Flask app for Cloud Run
app = Flask(__name__)

# Enable CORS for Salesforce (Lightning Web Components) and Apex callouts
# For production, you may want to restrict origins to your Salesforce org
CORS(app, resources={
    r"/*": {
        "origins": "*",  # In production, replace with your Salesforce domain
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})


@app.route('/', methods=['POST', 'GET'])
def handle_request():
    """
    HTTP Cloud Function entry point.

    Expected request body:
    {
        "task": "Go to Google and search for salesforce"
    }

    Returns:
    {
        "status": "completed",
        "final_url": "https://www.google.com/search?q=salesforce",
        "final_response": "I have completed the search...",
        "actions_taken": [...]
    }
    """

    # Handle GET request (health check)
    if request.method == 'GET':
        return jsonify({"status": "healthy", "service": "gemini-browser-agent"}), 200

    # Handle POST request
    data = request.get_json()
    task = data.get('task', '')

    if not task:
        return jsonify({"error": "No task provided"}), 400

    try:
        result = execute_browser_task(task)
        return jsonify(result), 200
    except Exception as e:
        logger.exception("Error executing browser task")
        return jsonify({"error": str(e)}), 500


@app.route('/diagram', methods=['POST'])
def handle_diagram():
    """
    Create a Lucidchart architecture diagram for Salesforce opportunity.
    
    Expected request body:
    {
        "opportunity_id": "006abc",
        "company_name": "Acme Financial Services",
        "industry": "Financial Services",
        "products": ["Sales Cloud", "Service Cloud", "MuleSoft", "Tableau"]
    }
    
    Returns:
    {
        "status": "completed",
        "lucidchart_url": "https://lucid.app/documents/view/...",
        "screenshot_b64": "iVBORw0KGgo...",
        "message": "Created diagram with 4 components for Acme Financial Services",
        "actions_taken": [...]
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({
            "status": "error",
            "error": "No JSON data provided",
            "error_type": "validation_error"
        }), 400
    
    # Validate required fields
    required_fields = ['opportunity_id', 'company_name', 'products']
    missing_fields = [field for field in required_fields if not data.get(field)]
    
    if missing_fields:
        return jsonify({
            "status": "error",
            "error": f"Missing required fields: {', '.join(missing_fields)}",
            "error_type": "validation_error"
        }), 400
    
    # Validate products is a list
    if not isinstance(data.get('products'), list) or len(data.get('products', [])) == 0:
        return jsonify({
            "status": "error",
            "error": "products must be a non-empty list",
            "error_type": "validation_error"
        }), 400
    
    try:
        # Build task instructions
        task_instructions = build_diagram_task_instructions(data)
        
        # Execute browser task
        result = execute_browser_task(task_instructions)
        
        # Check if task completed successfully
        if result.get("status") != "completed":
            return jsonify({
                "status": "error",
                "error": result.get("final_response", "Task did not complete successfully"),
                "error_type": "execution_error",
                "actions_taken": result.get("actions_taken", [])
            }), 500
        
        # Extract Lucidchart URL from final_response or final_url
        lucidchart_url = result.get("final_url", "")
        final_response = result.get("final_response", "")
        
        # Try to extract URL from final_response if it contains "lucid.app"
        if "lucid.app" in final_response:
            url_match = re.search(r'https?://[^\s]*lucid\.app[^\s]*', final_response)
            if url_match:
                lucidchart_url = url_match.group(0)
        
        # If we still don't have a Lucidchart URL, use final_url if it's a Lucidchart URL
        if not lucidchart_url or "lucid.app" not in lucidchart_url:
            if "lucid.app" in result.get("final_url", ""):
                lucidchart_url = result.get("final_url")
            else:
                # Fallback: use final_url anyway
                lucidchart_url = result.get("final_url", "")
        
        # Build response message
        company_name = data.get('company_name', 'Company')
        product_count = len(data.get('products', []))
        message = f"Created diagram with {product_count} component{'s' if product_count != 1 else ''} for {company_name}"
        
        return jsonify({
            "status": "completed",
            "lucidchart_url": lucidchart_url,
            "screenshot_b64": result.get("final_screenshot_b64", ""),
            "message": message,
            "actions_taken": result.get("actions_taken", [])
        }), 200
        
    except ValueError as e:
        logger.exception("Validation error in diagram endpoint")
        return jsonify({
            "status": "error",
            "error": str(e),
            "error_type": "validation_error"
        }), 400
    except Exception as e:
        logger.exception("Error executing diagram task")
        return jsonify({
            "status": "error",
            "error": str(e),
            "error_type": "execution_error"
        }), 500


@app.route('/roi', methods=['POST'])
def handle_roi():
    """
    Populate Google Sheets ROI calculator and extract calculated values.
    
    Expected request body:
    {
        "opportunity_id": "006abc",
        "company_name": "Acme Corp",
        "total_initial_investment_cost": 500000,
        "average_annual_cash_flow": 200000,
        "annual_profit": 50000000,
        "roi_sheet_url": "https://docs.google.com/spreadsheets/d/TEMPLATE_ID"
    }
    
    Returns:
    {
        "status": "completed",
        "sheet_url": "https://docs.google.com/spreadsheets/...",
        "roi_percentage": 412,
        "payback_months": 8,
        "annual_savings": 847000,
        "screenshot_b64": "iVBORw0KGgo...",
        "message": "ROI model shows 412% five-year return for Acme Corp",
        "actions_taken": [...]
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({
            "status": "error",
            "error": "No JSON data provided",
            "error_type": "validation_error"
        }), 400
    
    # Validate required fields
    required_fields = ['opportunity_id', 'company_name', 'total_initial_investment_cost',
                       'average_annual_cash_flow', 'roi_sheet_url']
    missing_fields = [field for field in required_fields if not data.get(field)]
    
    if missing_fields:
        return jsonify({
            "status": "error",
            "error": f"Missing required fields: {', '.join(missing_fields)}",
            "error_type": "validation_error"
        }), 400
    
    # Validate numeric fields (required for ROI calculation)
    try:
        total_initial_investment_cost = float(data.get('total_initial_investment_cost', 0))
        average_annual_cash_flow = float(data.get('average_annual_cash_flow', 0))
        if total_initial_investment_cost < 0 or average_annual_cash_flow <= 0:
            raise ValueError("total_initial_investment_cost must be >= 0 and average_annual_cash_flow must be > 0")
    except (ValueError, TypeError):
        return jsonify({
            "status": "error",
            "error": "total_initial_investment_cost and average_annual_cash_flow must be valid numbers (investment >= 0, cash flow > 0)",
            "error_type": "validation_error"
        }), 400
    
    # Validate optional numeric fields (for demo completeness)
    try:
        if data.get('annual_profit') is not None:
            annual_profit = float(data.get('annual_profit', 0))
            if annual_profit < 0:
                raise ValueError("annual_profit must be >= 0")
        if data.get('employee_count') is not None:
            employee_count = int(data.get('employee_count', 0))
            if employee_count < 0:
                raise ValueError("employee_count must be >= 0")
    except (ValueError, TypeError) as e:
        return jsonify({
            "status": "error",
            "error": f"Invalid optional field: {str(e)}",
            "error_type": "validation_error"
        }), 400
    
    try:
        # Build task instructions
        task_instructions = build_roi_task_instructions(data)
        
        # Execute browser task
        result = execute_browser_task(task_instructions)
        
        # Log the result for debugging
        logger.info(f"Task result status: {result.get('status')}")
        logger.info(f"Final response preview: {result.get('final_response', '')[:200]}")
        
        # Check if task completed successfully
        if result.get("status") != "completed":
            error_msg = result.get("final_response", "Task did not complete successfully")
            logger.warning(f"Task did not complete. Status: {result.get('status')}, Response: {error_msg[:200]}")
            return jsonify({
                "status": "error",
                "error": error_msg if error_msg else "Task did not complete successfully. The spreadsheet may have been filled, but Gemini did not return a completion response.",
                "error_type": "execution_error",
                "actions_taken": result.get("actions_taken", []),
                "final_response_preview": result.get("final_response", "")[:500] if result.get("final_response") else None
            }), 500
        
        # Extract ROI values from final_response
        try:
            roi_data = extract_structured_data_from_response(result.get("final_response", ""))
            logger.info(f"Successfully extracted ROI data: {roi_data}")
        except ValueError as e:
            logger.warning(f"Could not extract ROI values: {e}")
            logger.warning(f"Full final_response: {result.get('final_response', '')}")
            # Return partial result if we can't extract values
            return jsonify({
                "status": "error",
                "error": f"Could not extract ROI values from response: {str(e)}",
                "error_type": "execution_error",
                "final_response": result.get("final_response", "")[:1000],
                "actions_taken": result.get("actions_taken", [])
            }), 500
        
        # Get sheet URL (use the input URL)
        sheet_url = data.get('roi_sheet_url', result.get("final_url", ""))
        
        # Build response message
        company_name = data.get('company_name', 'Company')
        roi_pct = roi_data.get('roi_percentage', 0)
        message = f"ROI model shows {roi_pct}% five-year return for {company_name}"
        
        return jsonify({
            "status": "completed",
            "sheet_url": sheet_url,
            "roi_percentage": roi_data.get('roi_percentage'),
            "payback_months": roi_data.get('payback_months'),
            "annual_savings": roi_data.get('annual_savings'),
            "screenshot_b64": result.get("final_screenshot_b64", ""),
            "message": message,
            "actions_taken": result.get("actions_taken", [])
        }), 200
        
    except ValueError as e:
        logger.exception("Validation error in ROI endpoint")
        return jsonify({
            "status": "error",
            "error": str(e),
            "error_type": "validation_error"
        }), 400
    except Exception as e:
        logger.exception("Error executing ROI task")
        return jsonify({
            "status": "error",
            "error": str(e),
            "error_type": "execution_error"
        }), 500


if __name__ == '__main__':
    # Run Flask app locally for development
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

