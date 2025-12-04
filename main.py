from flask import Flask, jsonify, request
from google import genai
from google.genai import types
from google.genai.types import FunctionResponse, FinishReason
from playwright.sync_api import sync_playwright
import base64
import time
import os
import logging

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

        # Agent loop - max 10 turns for safety
        for turn in range(10):
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


# Create Flask app for Cloud Run
app = Flask(__name__)


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
