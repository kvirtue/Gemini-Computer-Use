#!/usr/bin/env python3
"""
Local test script for Gemini Computer Use with VISIBLE browser.
Run this to see the browser automation in action on your screen.

Usage:
    cd "/Users/kvirtue/Documents/Gemini Computer Use"
    source venv/bin/activate
    python local_test.py "Go to wikipedia.org and search for Salesforce"
"""

import sys
import os
import time
import base64
from google import genai
from google.genai import types
from google.genai.types import FunctionResponse
from playwright.sync_api import sync_playwright

# Screen dimensions
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900

# Your API key - set this or use environment variable
API_KEY = os.environ.get('GEMINI_API_KEY', 'YOUR_API_KEY_HERE')

# Playwright key mapping
PLAYWRIGHT_KEY_MAP = {
    "control": "Control", "shift": "Shift", "alt": "Alt",
    "enter": "Enter", "tab": "Tab", "escape": "Escape",
    "backspace": "Backspace", "delete": "Delete",
    "pageup": "PageUp", "pagedown": "PageDown",
    "command": "Meta", "meta": "Meta",
}


def denormalize_x(x): return int(x / 1000 * SCREEN_WIDTH)
def denormalize_y(y): return int(y / 1000 * SCREEN_HEIGHT)


def execute_key_combination(page, keys_str):
    keys = keys_str.split("+")
    keys = [PLAYWRIGHT_KEY_MAP.get(k.lower(), k) for k in keys]
    for key in keys[:-1]:
        page.keyboard.down(key)
    page.keyboard.press(keys[-1])
    for key in reversed(keys[:-1]):
        page.keyboard.up(key)


def run_local_agent(task: str, initial_url: str = "https://www.google.com"):
    """Run the browser agent locally with visible browser."""
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting Gemini Computer Use Agent")
    print(f"📋 Task: {task}")
    print(f"🌐 Starting URL: {initial_url}")
    print(f"{'='*60}\n")

    # Initialize Gemini client
    client = genai.Client(api_key=API_KEY)

    # Start Playwright with VISIBLE browser
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(
        headless=False,  # <-- VISIBLE BROWSER!
        slow_mo=500,     # Slow down actions so you can see them
    )
    context = browser.new_context(
        viewport={"width": SCREEN_WIDTH, "height": SCREEN_HEIGHT}
    )
    page = context.new_page()

    try:
        # Navigate to starting page
        print(f"📍 Navigating to {initial_url}...")
        page.goto(initial_url)
        page.wait_for_load_state()
        time.sleep(1)

        # Configure Gemini
        config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=8192,
            tools=[types.Tool(
                computer_use=types.ComputerUse(
                    environment=types.Environment.ENVIRONMENT_BROWSER
                )
            )],
        )

        # Take initial screenshot
        screenshot = page.screenshot(type="png")
        
        # Save screenshot for reference
        with open("screenshot_0_initial.png", "wb") as f:
            f.write(screenshot)
        print("📸 Saved: screenshot_0_initial.png")

        # Build conversation
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(text=task),
                    types.Part.from_bytes(data=screenshot, mime_type='image/png')
                ]
            )
        ]

        # Agent loop
        for turn in range(10):
            print(f"\n--- Turn {turn + 1} ---")
            print("🤔 Asking Gemini what to do next...")

            response = client.models.generate_content(
                model='gemini-2.5-computer-use-preview-10-2025',
                contents=contents,
                config=config,
            )

            if not response.candidates:
                print("❌ No response from model")
                break

            candidate = response.candidates[0]
            if candidate.content:
                contents.append(candidate.content)

            # Check for function calls
            has_function_calls = False
            if candidate.content and candidate.content.parts:
                has_function_calls = any(p.function_call for p in candidate.content.parts)

            if not has_function_calls:
                # Extract text response
                text = " ".join([p.text for p in candidate.content.parts if p.text])
                print(f"\n✅ Task Complete!")
                print(f"📝 Response: {text}")
                break

            # Execute function calls
            function_responses = []
            for part in candidate.content.parts:
                if part.function_call:
                    fc = part.function_call
                    action = fc.name
                    args = fc.args or {}

                    print(f"▶️  Action: {action}")
                    if args:
                        print(f"   Args: {dict(args)}")

                    # Execute the action
                    try:
                        if action == "click_at":
                            x, y = denormalize_x(args["x"]), denormalize_y(args["y"])
                            print(f"   📍 Clicking at ({x}, {y})")
                            page.mouse.click(x, y)

                        elif action == "type_text_at":
                            x, y = denormalize_x(args["x"]), denormalize_y(args["y"])
                            text = args["text"]
                            print(f"   ⌨️  Typing: '{text}' at ({x}, {y})")
                            page.mouse.click(x, y)
                            if args.get("clear_before_typing", True):
                                execute_key_combination(page, "Control+A")
                                execute_key_combination(page, "Delete")
                            page.keyboard.type(text)
                            if args.get("press_enter", False):
                                page.keyboard.press("Enter")

                        elif action == "scroll_document":
                            direction = args["direction"]
                            print(f"   📜 Scrolling {direction}")
                            if direction == "down":
                                page.keyboard.press("PageDown")
                            elif direction == "up":
                                page.keyboard.press("PageUp")

                        elif action == "navigate":
                            url = args["url"]
                            if not url.startswith(("http://", "https://")):
                                url = "https://" + url
                            print(f"   🌐 Navigating to {url}")
                            page.goto(url)

                        elif action == "hover_at":
                            x, y = denormalize_x(args["x"]), denormalize_y(args["y"])
                            print(f"   🖱️  Hovering at ({x}, {y})")
                            page.mouse.move(x, y)

                        elif action == "go_back":
                            print("   ⬅️  Going back")
                            page.go_back()

                        elif action == "search":
                            print("   🔍 Going to search engine")
                            page.goto("https://www.google.com")

                        elif action == "wait_5_seconds":
                            print("   ⏳ Waiting 5 seconds...")
                            time.sleep(5)

                        elif action == "key_combination":
                            keys = args["keys"]
                            print(f"   ⌨️  Pressing {keys}")
                            execute_key_combination(page, keys)

                        else:
                            print(f"   ⚠️  Unknown action: {action}")

                    except Exception as e:
                        print(f"   ❌ Error: {e}")

                    # Wait and take screenshot
                    page.wait_for_load_state()
                    time.sleep(1)
                    
                    new_screenshot = page.screenshot(type="png")
                    
                    # Save screenshot
                    filename = f"screenshot_{turn + 1}_{action}.png"
                    with open(filename, "wb") as f:
                        f.write(new_screenshot)
                    print(f"   📸 Saved: {filename}")

                    function_responses.append(
                        FunctionResponse(
                            name=action,
                            response={"url": page.url},
                            parts=[types.FunctionResponsePart(
                                inline_data=types.FunctionResponseBlob(
                                    mime_type="image/png",
                                    data=new_screenshot
                                )
                            )]
                        )
                    )

            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(function_response=fr) for fr in function_responses]
                )
            )

        print(f"\n{'='*60}")
        print(f"🏁 Final URL: {page.url}")
        print(f"{'='*60}")
        
        # Keep browser open for inspection
        input("\n⏸️  Press Enter to close the browser...")

    finally:
        browser.close()
        playwright.stop()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default task
        task = "Go to wikipedia.org and search for Salesforce"
    else:
        task = " ".join(sys.argv[1:])
    
    # Set your API key here if not in environment
    if API_KEY == 'YOUR_API_KEY_HERE':
        print("⚠️  Please set GEMINI_API_KEY environment variable or edit API_KEY in this script")
        print("   export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)
    
    run_local_agent(task, initial_url="https://www.google.com")

