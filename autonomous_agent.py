#!/usr/bin/env python3
"""
Autonomous Agent - A fully autonomous AI agent that can plan and execute complex goals
by creating and using tools dynamically.

This agent operates in phases:
1. Setup: Get API key and goal, create tools directory
2. Planning: Break down goal into executable steps
3. Execution: Iterate through steps, discovering/creating tools as needed
4. Completion: Report success and terminate
"""

import os
import importlib
import sys
import getpass
import re
import requests
import json


def run_agent():
    """Main function that orchestrates the autonomous agent workflow."""
    
    # Phase 1: Setup
    print("🤖 Autonomous Agent Initializing...")
    print("=" * 50)
    
    # Get Gemini API key securely
    api_key = getpass.getpass("Enter your Gemini API key: ")
    if not api_key:
        print("❌ API key is required. Exiting.")
        return
    
    # Get user's high-level goal
    print("\n🎯 What would you like me to accomplish?")
    print("Example: 'Summarize the top 5 AI news articles from today and save them to a file named news.txt'")
    goal = input("Your goal: ").strip()
    if not goal:
        print("❌ Goal is required. Exiting.")
        return
    
    # Create tools directory and add to path
    tools_dir = "tools"
    if not os.path.exists(tools_dir):
        os.makedirs(tools_dir)
        print(f"📁 Created {tools_dir}/ directory")
    
    # Add tools directory to Python path for dynamic imports
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    
    # Initialize Gemini API configuration
    try:
        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        gemini_headers = {
            'Content-Type': 'application/json',
            'X-goog-api-key': api_key
        }
        print("✅ Gemini API configuration initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Gemini API configuration: {e}")
        return
    
    # Phase 2: Planning
    print(f"\n🧠 Planning how to achieve: {goal}")
    print("-" * 50)
    
    planning_system_prompt = """You are a world-class planner AI. Your job is to break down a high-level user goal into a sequence of small, atomic, and logical steps. Each step must have a clear sub-goal. The output must be a numbered list of these steps. Do not add any conversational fluff."""
    
    try:
        response = requests.post(
            gemini_url,
            headers=gemini_headers,
            json={
                "contents": [
                    {
                        "parts": [
                            {
                                "text": f"System: {planning_system_prompt}\n\nUser: {goal}"
                            }
                        ]
                    }
                ]
            }
        )
        response.raise_for_status()
        response_data = response.json()
        plan_text = response_data['candidates'][0]['content']['parts'][0]['text']
        print("📋 Generated plan:")
        print(plan_text)
        
        # Parse the plan into a list of steps
        steps = parse_plan(plan_text)
        if not steps:
            print("❌ Failed to parse plan into executable steps. Exiting.")
            return
        
        print(f"\n📊 Plan parsed into {len(steps)} steps")
        
    except Exception as e:
        print(f"❌ Failed to generate plan: {e}")
        return
    
    # Phase 3: Execution Loop
    print(f"\n🚀 Starting execution of {len(steps)} steps...")
    print("=" * 50)
    
    # Initialize execution context (short-term memory)
    execution_context = {}
    
    for i, step in enumerate(steps, 1):
        print(f"\n[Step {i}/{len(steps)}] {step}")
        print("-" * 30)
        
        try:
            # Tool Discovery: Check if suitable tool exists
            existing_tool = discover_existing_tool(gemini_url, gemini_headers, step, tools_dir)
            
            if existing_tool:
                print(f"🔍 Found existing tool: {existing_tool}")
                result = execute_existing_tool(existing_tool, step, execution_context)
            else:
                print(f"🛠️  No suitable tool found. Creating new tool...")
                tool_filename = generate_tool_filename(step)
                result = create_and_execute_tool(gemini_url, gemini_headers, step, tool_filename, tools_dir, execution_context)
            
            # Store result in execution context for next steps
            execution_context[f"step_{i}_result"] = result
            print(f"✅ Step {i} completed successfully")
            
        except Exception as e:
            print(f"❌ Step {i} failed: {e}")
            print("🔄 Continuing with next step...")
            continue
    
    # Phase 4: Completion
    print("\n" + "=" * 50)
    print("🎉 Goal achieved successfully!")
    print("=" * 50)


def parse_plan(plan_text):
    """Parse the LLM-generated plan into a list of executable steps."""
    lines = plan_text.strip().split('\n')
    steps = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove numbering and common prefixes
        step = re.sub(r'^\d+\.?\s*', '', line)
        step = re.sub(r'^[-*]\s*', '', step)
        
        if step and len(step) > 5:  # Filter out very short lines
            steps.append(step)
    
    return steps


def discover_existing_tool(gemini_url, gemini_headers, step, tools_dir):
    """Check if any existing tool can handle the current step."""
    if not os.path.exists(tools_dir):
        return None
    
    # Get list of available tools with their descriptions
    available_tools = []
    for filename in os.listdir(tools_dir):
        if filename.endswith('.py') and filename != '__init__.py':
            filepath = os.path.join(tools_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    first_line = f.readline().strip()
                    # Extract docstring if it's a comment
                    if first_line.startswith('"""') or first_line.startswith("'''"):
                        # Read until end of docstring
                        docstring = ""
                        for line in f:
                            docstring += line
                            if '"""' in line or "'''" in line:
                                break
                        available_tools.append(f"{filename}: {docstring.strip()}")
                    else:
                        available_tools.append(f"{filename}: {first_line}")
            except Exception:
                available_tools.append(f"{filename}: No description available")
    
    if not available_tools:
        return None
    
    # Use LLM to determine if any tool is suitable
    discovery_system_prompt = """You are a tool-matching AI. Given a task and a list of available tools with their descriptions, determine if any tool can perform the task. Respond with the exact filename of the best tool (e.g., 'web_scraper.py') or 'None' if no tool is suitable."""
    
    user_prompt = f"Task: {step}\n\nAvailable Tools:\n" + "\n".join(available_tools)
    
    try:
        response = requests.post(
            gemini_url,
            headers=gemini_headers,
            json={
                "contents": [
                    {
                        "parts": [
                            {
                                "text": f"System: {discovery_system_prompt}\n\nUser: {user_prompt}"
                            }
                        ]
                    }
                ]
            }
        )
        response.raise_for_status()
        response_data = response.json()
        result = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
        
        # Clean up the result
        if result.lower() == 'none' or 'none' in result.lower():
            return None
        
        # Extract filename if it's wrapped in quotes or has extra text
        filename_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*\.py)', result)
        if filename_match:
            filename = filename_match.group(1)
            if os.path.exists(os.path.join(tools_dir, filename)):
                return filename
        
        return None
        
    except Exception as e:
        print(f"⚠️  Tool discovery failed: {e}")
        return None


def execute_existing_tool(tool_filename, step, execution_context):
    """Execute an existing tool to accomplish the current step."""
    try:
        # Remove .py extension for import
        module_name = tool_filename[:-3]
        
        # Import the module dynamically
        module = importlib.import_module(module_name)
        
        # Find the main function (assume it's the first function defined)
        main_function = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr) and not attr_name.startswith('_'):
                main_function = attr
                break
        
        if not main_function:
            raise Exception(f"No executable function found in {tool_filename}")
        
        # Execute the function with context
        result = main_function(step, execution_context)
        return result
        
    except Exception as e:
        raise Exception(f"Failed to execute {tool_filename}: {e}")


def generate_tool_filename(step):
    """Generate a descriptive filename for a new tool based on the step."""
    # Clean the step text to create a filename
    filename = re.sub(r'[^a-zA-Z0-9\s]', '', step.lower())
    filename = re.sub(r'\s+', '_', filename)
    filename = filename.strip('_')
    
    # Limit length and ensure it's descriptive
    if len(filename) > 30:
        filename = filename[:30]
    
    return f"{filename}.py"


def create_and_execute_tool(gemini_url, gemini_headers, step, tool_filename, tools_dir, execution_context):
    """Create a new tool and execute it to accomplish the current step."""
    
    # Generate the tool using LLM
    tool_creation_system_prompt = """You are a master Python programmer. Your task is to write a single, self-contained, and robust Python script that functions as a tool to accomplish a specific task. The script must contain a single function that takes necessary inputs and returns a result. It must include a detailed docstring explaining its purpose, arguments, and return value. The code must be clean, efficient, and include error handling. Do not include any example usage calls in the script, only the function definition and necessary imports."""
    
    user_prompt = f"Create a Python tool to accomplish this task: {step}. The tool should be named '{tool_filename}'."
    
    try:
        response = requests.post(
            gemini_url,
            headers=gemini_headers,
            json={
                "contents": [
                    {
                        "parts": [
                            {
                                "text": f"System: {tool_creation_system_prompt}\n\nUser: {user_prompt}"
                            }
                        ]
                    }
                ]
            }
        )
        response.raise_for_status()
        response_data = response.json()
        tool_code = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
        
        # Clean up the code (remove markdown formatting if present)
        if tool_code.startswith('```python'):
            tool_code = tool_code[9:]
        if tool_code.endswith('```'):
            tool_code = tool_code[:-3]
        
        tool_code = tool_code.strip()
        
        # Save the tool to file
        tool_filepath = os.path.join(tools_dir, tool_filename)
        with open(tool_filepath, 'w') as f:
            f.write(tool_code)
        
        print(f"📝 Created tool: {tool_filename}")
        
        # Check for required dependencies
        check_dependencies(tool_code)
        
        # Execute the newly created tool
        return execute_existing_tool(tool_filename, step, execution_context)
        
    except Exception as e:
        raise Exception(f"Failed to create tool {tool_filename}: {e}")


def check_dependencies(tool_code):
    """Check if the tool requires non-standard libraries and inform the user."""
    standard_libs = {
        'os', 'sys', 're', 'json', 'datetime', 'time', 'random', 'math', 
        'collections', 'itertools', 'functools', 'pathlib', 'urllib', 'http',
        'ssl', 'socket', 'threading', 'subprocess', 'tempfile', 'shutil', 'requests'
    }
    
    # Simple regex to find import statements
    import_pattern = r'^(?:from\s+(\w+)|import\s+(\w+))'
    
    for line in tool_code.split('\n'):
        line = line.strip()
        if line.startswith('import ') or line.startswith('from '):
            match = re.match(import_pattern, line)
            if match:
                lib_name = match.group(1) or match.group(2)
                if lib_name not in standard_libs:
                    print(f"⚠️  Tool requires the '{lib_name}' library. Please run: pip install {lib_name}")


if __name__ == "__main__":
    try:
        run_agent()
    except KeyboardInterrupt:
        print("\n\n⏹️  Agent execution interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()