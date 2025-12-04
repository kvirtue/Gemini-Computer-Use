# Gemini 2.5 Computer Use + Salesforce Agentforce Integration

## Project Overview

Build a middleware service that enables Salesforce Agentforce agents to perform browser automation using Google's Gemini 2.5 Computer Use model.

### What We're Building

1. **Google Cloud Function** - A serverless Python function that:

   - Receives browser task requests from Salesforce
   - Controls a headless browser using Playwright
   - Uses Gemini 2.5 Computer Use to "see" screenshots and decide what actions to take
   - Returns results back to Salesforce

2. **Salesforce Apex Invocable Action** - Connects Agentforce to the Cloud Function

3. **Agentforce Agent Configuration** - Enables natural language browser automation

### End Result

A user can tell an Agentforce agent: "Search Google for Salesforce" and the agent will:

1. Call the Cloud Function
2. The function opens a browser, navigates to Google
3. Gemini analyzes the screenshot and types the search query
4. Returns the search results to the user

---

## 🚦 CURRENT PROGRESS & STATUS

### ✅ Completed Steps

| Step | Description                                                                                | Status  |
| ---- | ------------------------------------------------------------------------------------------ | ------- |
| 1    | Created Google Cloud account                                                               | ✅ Done |
| 2    | Created GCP project with billing enabled                                                   | ✅ Done |
| 3    | Enabled required APIs (Vertex AI, Cloud Run, Secret Manager, Cloud Build, Cloud Functions) | ✅ Done |
| 4    | Installed gcloud CLI on Mac via Homebrew                                                   | ✅ Done |
| 5    | Authenticated gcloud with `gcloud init`                                                    | ✅ Done |
| 6    | Verified project is set correctly (`gcloud config get-value project`)                      | ✅ Done |
| 7    | Created project folder "Gemini Computer Use" in Cursor                                     | ✅ Done |

### 🔴 CURRENTLY STUCK

**Problem:** Need to create and activate Python virtual environment

**Current Directory Contents:**

```
Gemini Computer Use/
├── .cursor/
├── computer-use.md.md
└── Supported Models _ Models and Prompts _ Agentforce Developer Guide _ Salesforce Developers.pdf
```

**What needs to happen:**

```bash
# 1. Create the virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Should see (venv) in prompt:
# (venv) kvirtue@kvirtue-ltmvye2 Gemini Computer Use %
```

**Potential issues to check:**

- Is Python 3.10+ installed? Run: `python3 --version`
- If Python not found, install with: `brew install python@3.11`

### 📋 Remaining Steps (After venv is working)

| Step | Description                               | Status        |
| ---- | ----------------------------------------- | ------------- |
| 8    | Activate Python virtual environment       | 🔴 Stuck here |
| 9    | Create `main.py` with Cloud Function code | ⏳ Pending    |
| 10   | Create `requirements.txt`                 | ⏳ Pending    |
| 11   | Store API key in Google Secret Manager    | ⏳ Pending    |
| 12   | Deploy Cloud Function to GCP              | ⏳ Pending    |
| 13   | Test Cloud Function with curl             | ⏳ Pending    |
| 14   | Create Named Credential in Salesforce     | ⏳ Pending    |
| 15   | Create Apex Invocable Action              | ⏳ Pending    |
| 16   | Configure Agentforce Agent                | ⏳ Pending    |
| 17   | End-to-end test                           | ⏳ Pending    |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SALESFORCE ORG                               │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Agentforce  │───▶│    Flow      │───▶│   Apex Invocable     │  │
│  │    Agent     │    │   (Trigger)  │    │      Action          │  │
│  └──────────────┘    └──────────────┘    └───────────┬───────────┘  │
└──────────────────────────────────────────────────────┼───────────────┘
                                                       │ HTTP Callout
                                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GOOGLE CLOUD FUNCTION                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Cloud Run Function (Python)                     │    │
│  │  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │    │
│  │  │  Playwright │◀──▶│  Agent Loop  │◀──▶│  Gemini 2.5    │  │    │
│  │  │  (Browser)  │    │  Controller  │    │  Computer Use  │  │    │
│  │  └─────────────┘    └──────────────┘    └────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component              | Technology                                                          | Purpose                                    |
| ---------------------- | ------------------------------------------------------------------- | ------------------------------------------ |
| AI Model               | Gemini 2.5 Computer Use (`gemini-2.5-computer-use-preview-10-2025`) | Analyzes screenshots, generates UI actions |
| Browser Automation     | Playwright (Python)                                                 | Executes clicks, typing, navigation        |
| Serverless Compute     | Google Cloud Functions (Gen 2)                                      | Hosts the middleware                       |
| Secret Storage         | Google Secret Manager                                               | Stores API keys securely                   |
| Salesforce Integration | Apex Invocable Action                                               | Callable from Agentforce/Flows             |

---

## File Structure (Target)

```
Gemini Computer Use/
├── main.py              # Cloud Function entry point (TO CREATE)
├── requirements.txt     # Python dependencies (TO CREATE)
├── venv/               # Python virtual environment (TO CREATE)
├── computer-use.md.md   # Reference documentation (EXISTS)
└── .cursor/            # Cursor settings (EXISTS)
```

---

## Part 1: Cloud Function Code

### File: `main.py`

```python
import functions_framework
from flask import jsonify, request
from google import genai
from google.genai import types
from playwright.sync_api import sync_playwright
import base64
import time

# Initialize Gemini client
client = genai.Client()

# Screen dimensions for the browser
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900


def denormalize_x(x: int) -> int:
    """
    Convert normalized x coordinate (0-999) to actual pixel coordinate.
    Gemini outputs coordinates on a 0-999 scale regardless of screen size.
    """
    return int(x / 1000 * SCREEN_WIDTH)


def denormalize_y(y: int) -> int:
    """
    Convert normalized y coordinate (0-999) to actual pixel coordinate.
    Gemini outputs coordinates on a 0-999 scale regardless of screen size.
    """
    return int(y / 1000 * SCREEN_HEIGHT)


def execute_browser_task(task_description: str) -> dict:
    """
    Execute a browser task using Gemini Computer Use.

    Args:
        task_description: Natural language description of what to do
                         Example: "Go to Google and search for salesforce"

    Returns:
        dict with status, final_url, actions_taken, and final_response
    """

    # Start Playwright browser
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(
        viewport={"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT}
    )
    page = context.new_page()

    try:
        # Navigate to starting page
        page.goto("https://www.google.com")

        # Configure Gemini Computer Use
        config = types.GenerateContentConfig(
            tools=[types.Tool(
                computer_use=types.ComputerUse(
                    environment=types.Environment.ENVIRONMENT_BROWSER
                )
            )],
        )

        # Take initial screenshot
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

            # Ask Gemini what to do next
            response = client.models.generate_content(
                model='gemini-2.5-computer-use-preview-10-2025',
                contents=contents,
                config=config,
            )

            candidate = response.candidates[0]
            contents.append(candidate.content)

            # Check if Gemini is done (no more function calls)
            has_function_calls = any(
                part.function_call for part in candidate.content.parts
            )

            if not has_function_calls:
                # Extract final text response
                text_response = " ".join([
                    part.text for part in candidate.content.parts if part.text
                ])
                result["status"] = "completed"
                result["final_response"] = text_response
                break

            # Execute each function call from Gemini
            function_responses = []

            for part in candidate.content.parts:
                if part.function_call:
                    fc = part.function_call
                    action_name = fc.name
                    args = fc.args

                    # Log the action
                    result["actions_taken"].append({
                        "action": action_name,
                        "args": dict(args)
                    })

                    # Execute the action based on type
                    if action_name == "click_at":
                        x = denormalize_x(args["x"])
                        y = denormalize_y(args["y"])
                        page.mouse.click(x, y)

                    elif action_name == "type_text_at":
                        x = denormalize_x(args["x"])
                        y = denormalize_y(args["y"])
                        page.mouse.click(x, y)
                        # Clear existing text
                        page.keyboard.press("Meta+A")
                        page.keyboard.press("Backspace")
                        # Type new text
                        page.keyboard.type(args["text"])
                        if args.get("press_enter", False):
                            page.keyboard.press("Enter")

                    elif action_name == "scroll_document":
                        direction = args["direction"]
                        if direction == "down":
                            page.mouse.wheel(0, 300)
                        elif direction == "up":
                            page.mouse.wheel(0, -300)

                    elif action_name == "navigate":
                        page.goto(args["url"])

                    elif action_name == "go_back":
                        page.go_back()

                    elif action_name == "wait_5_seconds":
                        time.sleep(5)

                    # Wait for page to settle after action
                    time.sleep(1)
                    try:
                        page.wait_for_load_state(timeout=5000)
                    except:
                        pass

                    # Capture new screenshot after action
                    new_screenshot = page.screenshot(type="png")

                    # Build function response for Gemini
                    function_responses.append(
                        types.FunctionResponse(
                            name=action_name,
                            response={"url": page.url},
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

        # Capture final state
        final_screenshot = page.screenshot(type="png")
        result["final_screenshot_b64"] = base64.b64encode(final_screenshot).decode()
        result["final_url"] = page.url

        return result

    finally:
        browser.close()
        playwright.stop()


@functions_framework.http
def handle_request(request):
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

    # Verify API key (basic auth)
    api_key = request.headers.get('api-key')
    # TODO: Validate against Secret Manager in production

    if request.method == 'POST':
        data = request.get_json()
        task = data.get('task', '')

        if not task:
            return jsonify({"error": "No task provided"}), 400

        try:
            result = execute_browser_task(task)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Method not allowed"}), 405
```

### File: `requirements.txt`

```
functions-framework==3.*
google-genai
playwright
flask
```

---

## Part 2: Deployment Steps

### Prerequisites Completed

- [x] Google Cloud account created
- [x] GCP project created with billing enabled
- [x] APIs enabled: Vertex AI, Cloud Run, Secret Manager, Cloud Build, Cloud Functions
- [x] gcloud CLI installed and authenticated
- [ ] Python virtual environment set up ← **STUCK HERE**

### Step-by-Step After Virtual Environment

#### Step 1: Create the Files

After venv is working, create these two files in your project folder:

1. Create `main.py` with the Python code above
2. Create `requirements.txt` with the dependencies listed above

#### Step 2: Install Dependencies Locally (Optional, for testing)

```bash
# Make sure venv is activated first!
pip install -r requirements.txt
playwright install chromium
```

#### Step 3: Store API Key in Secret Manager

```bash
# Create a secret for your API key
echo -n "your-secure-api-key-here" | gcloud secrets create llm-connector-api-key --data-file=-

# Grant access to the Cloud Function service account (do this after deployment)
```

#### Step 4: Deploy the Cloud Function

```bash
# Navigate to your project folder
cd "Gemini Computer Use"

# Deploy the function
gcloud functions deploy gemini-browser-agent \
    --gen2 \
    --runtime=python310 \
    --region=us-central1 \
    --source=. \
    --entry-point=handle_request \
    --trigger-http \
    --allow-unauthenticated \
    --memory=2048MB \
    --timeout=300s
```

#### Step 5: Get the Function URL

```bash
gcloud functions describe gemini-browser-agent \
    --region=us-central1 \
    --format='value(serviceConfig.uri)'
```

#### Step 6: Test with curl

```bash
curl -X POST \
  YOUR_FUNCTION_URL \
  -H "Content-Type: application/json" \
  -H "api-key: your-api-key" \
  -d '{"task": "Go to Google and search for salesforce"}'
```

---

## Part 3: Salesforce Integration

### File: Apex Class `GeminiBrowserAction.cls`

```apex
/**
 * Invocable action for Agentforce to execute browser tasks
 * using Gemini 2.5 Computer Use
 */
public class GeminiBrowserAction {

    public class BrowserTaskRequest {
        @InvocableVariable(required=true label='Task Description')
        public String taskDescription;
    }

    public class BrowserTaskResponse {
        @InvocableVariable(label='Status')
        public String status;

        @InvocableVariable(label='Final URL')
        public String finalUrl;

        @InvocableVariable(label='Result Summary')
        public String resultSummary;

        @InvocableVariable(label='Actions Taken')
        public String actionsTaken;

        @InvocableVariable(label='Error Message')
        public String errorMessage;
    }

    @InvocableMethod(
        label='Execute Browser Task with Gemini'
        description='Uses Gemini 2.5 Computer Use to perform browser automation'
        category='Browser Automation'
    )
    public static List<BrowserTaskResponse> executeBrowserTask(
        List<BrowserTaskRequest> requests
    ) {
        List<BrowserTaskResponse> responses = new List<BrowserTaskResponse>();

        for (BrowserTaskRequest req : requests) {
            BrowserTaskResponse res = new BrowserTaskResponse();

            try {
                // Build request body
                Map<String, Object> requestBody = new Map<String, Object>{
                    'task' => req.taskDescription
                };

                // Make callout to Cloud Function
                HttpRequest httpReq = new HttpRequest();
                httpReq.setEndpoint('callout:Gemini_Browser_Agent');
                httpReq.setMethod('POST');
                httpReq.setHeader('Content-Type', 'application/json');
                httpReq.setBody(JSON.serialize(requestBody));
                httpReq.setTimeout(120000); // 2 minute timeout

                Http http = new Http();
                HttpResponse httpRes = http.send(httpReq);

                if (httpRes.getStatusCode() == 200) {
                    Map<String, Object> responseData =
                        (Map<String, Object>) JSON.deserializeUntyped(httpRes.getBody());

                    res.status = (String) responseData.get('status');
                    res.finalUrl = (String) responseData.get('final_url');
                    res.resultSummary = (String) responseData.get('final_response');

                    List<Object> actions =
                        (List<Object>) responseData.get('actions_taken');
                    if (actions != null) {
                        res.actionsTaken = JSON.serialize(actions);
                    }
                } else {
                    res.status = 'error';
                    res.errorMessage = 'HTTP ' + httpRes.getStatusCode() +
                        ': ' + httpRes.getBody();
                }

            } catch (Exception e) {
                res.status = 'error';
                res.errorMessage = e.getMessage();
            }

            responses.add(res);
        }

        return responses;
    }
}
```

### Salesforce Setup Steps

1. **Create Named Credential:**

   - Setup → Named Credentials → New
   - Label: `Gemini_Browser_Agent`
   - URL: `https://us-central1-YOUR_PROJECT.cloudfunctions.net/gemini-browser-agent`
   - Authentication: Custom Header with `api-key`

2. **Deploy Apex Class:**

   - Copy the Apex code above
   - Setup → Apex Classes → New

3. **Configure Agentforce:**
   - Setup → Agentforce → Agent Actions
   - Add the `Execute Browser Task with Gemini` action
   - Configure instructions for when to use it

---

## How Gemini Computer Use Works

### The Agent Loop

1. **Send screenshot + task to Gemini**
2. **Gemini analyzes and returns a function call** (e.g., `click_at`, `type_text_at`)
3. **Your code executes the action** in the browser
4. **Capture new screenshot**
5. **Send screenshot back to Gemini**
6. **Repeat until task is complete**

### Supported UI Actions

| Action            | Description          | Arguments                        |
| ----------------- | -------------------- | -------------------------------- |
| `click_at`        | Click at coordinates | `x`, `y` (0-999 normalized)      |
| `type_text_at`    | Click and type text  | `x`, `y`, `text`, `press_enter`  |
| `scroll_document` | Scroll the page      | `direction` (up/down/left/right) |
| `navigate`        | Go to URL            | `url`                            |
| `go_back`         | Browser back button  | none                             |
| `wait_5_seconds`  | Pause execution      | none                             |

### Coordinate System

Gemini outputs coordinates on a **normalized 0-999 scale**. Your code must convert them to actual pixels:

```python
actual_x = normalized_x / 1000 * screen_width
actual_y = normalized_y / 1000 * screen_height
```

---

## Security Considerations

1. **Run browser in headless mode** in a containerized environment
2. **Store API keys in Secret Manager**, not in code
3. **Implement rate limiting** on the Cloud Function
4. **Log all actions** for audit trails
5. **Consider domain allowlists** to restrict where the browser can navigate

---

## Testing Checklist

- [ ] Python virtual environment created and activated
- [ ] `main.py` and `requirements.txt` files created
- [ ] Cloud Function deploys successfully
- [ ] Function responds to HTTP POST requests
- [ ] Gemini model is accessible (no auth errors)
- [ ] Browser actions execute correctly
- [ ] Screenshots are captured and sent to Gemini
- [ ] Salesforce can reach the Cloud Function endpoint
- [ ] Apex action returns expected response format
- [ ] Agentforce agent triggers the action correctly

---

## Common Issues & Solutions

| Issue                                | Solution                                                      |
| ------------------------------------ | ------------------------------------------------------------- |
| `python3 -m venv venv` fails         | Install Python: `brew install python@3.11`                    |
| `source venv/bin/activate` not found | Run `python3 -m venv venv` first to create it                 |
| "Model not found"                    | Ensure you're using `gemini-2.5-computer-use-preview-10-2025` |
| Playwright fails to launch           | Increase Cloud Function memory to 2048MB+                     |
| Timeout errors                       | Increase timeout to 300s in deployment                        |
| Auth errors from Salesforce          | Check Named Credential configuration                          |
| Coordinates are wrong                | Verify denormalization math matches screen size               |

---

## Quick Commands Reference

```bash
# Check Python version
python3 --version

# Create virtual environment
python3 -m venv venv

# Activate virtual environment (Mac/Linux)
source venv/bin/activate

# Deactivate virtual environment
deactivate

# Check current GCP project
gcloud config get-value project

# Set GCP project
gcloud config set project YOUR_PROJECT_ID

# Deploy Cloud Function
gcloud functions deploy gemini-browser-agent \
    --gen2 \
    --runtime=python310 \
    --region=us-central1 \
    --source=. \
    --entry-point=handle_request \
    --trigger-http \
    --allow-unauthenticated \
    --memory=2048MB \
    --timeout=300s

# Get Cloud Function URL
gcloud functions describe gemini-browser-agent \
    --region=us-central1 \
    --format='value(serviceConfig.uri)'
```

---

## Next Steps After Basic Setup

1. Add more sophisticated error handling
2. Implement session persistence for multi-step tasks
3. Add screenshot storage to Salesforce Files
4. Create more specific agent topics for different use cases
5. Add monitoring and alerting
