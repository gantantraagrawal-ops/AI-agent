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
import time
import json
from together import Together


class RateLimiter:
    """Rate limiter to respect API limits (3 requests or 18k tokens per minute)."""
    
    def __init__(self, max_requests=3, max_tokens=18000, time_window=60):
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.time_window = time_window
        self.requests = []
        self.tokens_used = []
    
    def can_make_request(self, estimated_tokens=0):
        """Check if we can make a request without exceeding limits."""
        current_time = time.time()
        
        # Clean old entries
        self.requests = [t for t in self.requests if current_time - t < self.time_window]
        self.tokens_used = [t for t in self.tokens_used if current_time - t < self.time_window]
        
        # Check request limit
        if len(self.requests) >= self.max_requests:
            return False
        
        # Check token limit
        if sum(self.tokens_used) + estimated_tokens > self.max_tokens:
            return False
        
        return True
    
    def wait_if_needed(self, estimated_tokens=0):
        """Wait until we can make a request."""
        while not self.can_make_request(estimated_tokens):
            sleep_time = self.time_window - (time.time() - self.requests[0]) if self.requests else 1
            print(f"⏳ Rate limit reached. Waiting {sleep_time:.1f} seconds...")
            time.sleep(min(sleep_time, 5))  # Sleep in chunks to allow interruption
    
    def record_request(self, tokens_used):
        """Record a completed request."""
        current_time = time.time()
        self.requests.append(current_time)
        self.tokens_used.append(tokens_used)


class ResponseCache:
    """Simple cache for LLM responses to avoid duplicate API calls."""
    
    def __init__(self):
        self.cache = {}
    
    def get(self, key):
        """Get cached response if available."""
        return self.cache.get(key)
    
    def set(self, key, value):
        """Cache a response."""
        self.cache[key] = value
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()


def run_agent():
    """Main function that orchestrates the autonomous agent workflow."""
    
    # Phase 1: Setup
    print("🤖 Autonomous Agent Initializing...")
    print("=" * 50)
    
    # Get Together API key securely
    api_key = getpass.getpass("Enter your Together API key: ")
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
    
    # Initialize Together client and rate limiter
    try:
        client = Together(api_key=api_key)
        rate_limiter = RateLimiter()
        response_cache = ResponseCache()
        print("✅ Together API client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Together API client: {e}")
        return
    
    # Phase 2: Planning
    print(f"\n🧠 Planning how to achieve: {goal}")
    print("-" * 50)
    
    # Check cache first
    cache_key = f"plan:{goal}"
    cached_plan = response_cache.get(cache_key)
    
    if cached_plan:
        print("📋 Using cached plan")
        plan_text = cached_plan
    else:
        # Optimized planning prompt - shorter and more focused
        planning_system_prompt = """Break down this goal into numbered steps. Each step should be atomic and executable. Output only the numbered list, no extra text."""
        
        try:
            rate_limiter.wait_if_needed(estimated_tokens=1000)
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1",
                messages=[
                    {"role": "system", "content": planning_system_prompt},
                    {"role": "user", "content": goal}
                ],
                max_tokens=2048,  # Reduced from 4096
                temperature=0.1
            )
            plan_text = response.choices[0].message.content
            rate_limiter.record_request(estimated_tokens=1000)
            
            # Cache the plan
            response_cache.set(cache_key, plan_text)
            
            print("📋 Generated plan:")
            print(plan_text)
            
        except Exception as e:
            print(f"❌ Failed to generate plan: {e}")
            return
        
        # Parse the plan into a list of steps
        steps = parse_plan(plan_text)
        if not steps:
            print("❌ Failed to parse plan into executable steps. Exiting.")
            return
        
        print(f"\n📊 Plan parsed into {len(steps)} steps")
    
    # Phase 3: Execution Loop
    print(f"\n🚀 Starting execution of {len(steps)} steps...")
    print("=" * 50)
    
    # Initialize execution context (short-term memory)
    execution_context = {}
    
    # Batch tool discovery for efficiency
    print("🔍 Discovering existing tools...")
    existing_tools = discover_all_existing_tools(tools_dir)
    
    for i, step in enumerate(steps, 1):
        print(f"\n[Step {i}/{len(steps)}] {step}")
        print("-" * 30)
        
        try:
            # Tool Discovery: Check if suitable tool exists
            existing_tool = find_suitable_tool(step, existing_tools, client, rate_limiter, response_cache)
            
            if existing_tool:
                print(f"🔍 Found existing tool: {existing_tool}")
                result = execute_existing_tool(existing_tool, step, execution_context)
            else:
                print(f"🛠️  No suitable tool found. Creating new tool...")
                tool_filename = generate_tool_filename(step)
                result = create_and_execute_tool(client, step, tool_filename, tools_dir, execution_context, rate_limiter, response_cache)
            
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


def discover_all_existing_tools(tools_dir):
    """Discover all existing tools at once to avoid repeated API calls."""
    if not os.path.exists(tools_dir):
        return {}
    
    tools_info = {}
    for filename in os.listdir(tools_dir):
        if filename.endswith('.py') and filename != '__init__.py':
            filepath = os.path.join(tools_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    # Extract docstring
                    docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
                    if docstring_match:
                        description = docstring_match.group(1).strip()
                    else:
                        # Fallback to first line
                        f.seek(0)
                        first_line = f.readline().strip()
                        description = first_line
                    
                    tools_info[filename] = description
            except Exception:
                tools_info[filename] = "No description available"
    
    return tools_info


def find_suitable_tool(step, existing_tools, client, rate_limiter, response_cache):
    """Find a suitable tool from existing tools using cached LLM response."""
    if not existing_tools:
        return None
    
    # Check cache first
    cache_key = f"tool_match:{step}:{hash(str(existing_tools))}"
    cached_result = response_cache.get(cache_key)
    
    if cached_result:
        return cached_result if cached_result != "None" else None
    
    # Use LLM to determine if any tool is suitable
    # Optimized prompt - shorter and more focused
    discovery_system_prompt = """Given a task and available tools, respond with the exact filename (e.g., 'web_scraper.py') or 'None' if no tool fits."""
    
    # Create a concise list of tools
    tools_list = "\n".join([f"{f}: {d[:100]}..." for f, d in existing_tools.items()])
    user_prompt = f"Task: {step}\nTools:\n{tools_list}"
    
    try:
        rate_limiter.wait_if_needed(estimated_tokens=500)
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[
                {"role": "system", "content": discovery_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=50,  # Reduced from 100
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip()
        rate_limiter.record_request(estimated_tokens=500)
        
        # Clean up the result
        if result.lower() == 'none' or 'none' in result.lower():
            response_cache.set(cache_key, "None")
            return None
        
        # Extract filename if it's wrapped in quotes or has extra text
        filename_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*\.py)', result)
        if filename_match:
            filename = filename_match.group(1)
            if filename in existing_tools:
                response_cache.set(cache_key, filename)
                return filename
        
        response_cache.set(cache_key, "None")
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


def create_and_execute_tool(client, step, tool_filename, tools_dir, execution_context, rate_limiter, response_cache):
    """Create a new tool and execute it to accomplish the current step."""
    
    # Check cache first
    cache_key = f"tool_creation:{step}"
    cached_tool = response_cache.get(cache_key)
    
    if cached_tool:
        print(f"📝 Using cached tool: {tool_filename}")
        tool_code = cached_tool
    else:
        # Optimized tool creation prompt - shorter and more focused
        tool_creation_system_prompt = """Write a self-contained Python script with one function to accomplish the given task. Include docstring, error handling, and necessary imports. Output only the code."""
        
        user_prompt = f"Create a Python tool for: {step}"
        
        try:
            rate_limiter.wait_if_needed(estimated_tokens=3000)
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1",
                messages=[
                    {"role": "system", "content": tool_creation_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=3072,  # Reduced from 4096
                temperature=0.1
            )
            
            tool_code = response.choices[0].message.content.strip()
            rate_limiter.record_request(estimated_tokens=3000)
            
            # Clean up the code (remove markdown formatting if present)
            if tool_code.startswith('```python'):
                tool_code = tool_code[9:]
            if tool_code.endswith('```'):
                tool_code = tool_code[:-3]
            
            tool_code = tool_code.strip()
            
            # Cache the tool
            response_cache.set(cache_key, tool_code)
            
        except Exception as e:
            raise Exception(f"Failed to create tool {tool_filename}: {e}")
    
    # Save the tool to file
    tool_filepath = os.path.join(tools_dir, tool_filename)
    with open(tool_filepath, 'w') as f:
        f.write(tool_code)
    
    print(f"📝 Created tool: {tool_filename}")
    
    # Check for required dependencies
    check_dependencies(tool_code)
    
    # Execute the newly created tool
    return execute_existing_tool(tool_filename, step, execution_context)


def check_dependencies(tool_code):
    """Check if the tool requires non-standard libraries and inform the user."""
    standard_libs = {
        'os', 'sys', 're', 'json', 'datetime', 'time', 'random', 'math', 
        'collections', 'itertools', 'functools', 'pathlib', 'urllib', 'http',
        'ssl', 'socket', 'threading', 'subprocess', 'tempfile', 'shutil'
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