import os
import sys
import json
import ast
import glob
import importlib
import inspect
import logging
import subprocess
import re
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
AGENT_FILENAME = __file__
TOOLS_FILENAME = "tools.py"
TOGETHER_MODEL = "deepseek-ai/DeepSeek-R1"

class Agent:
    def __init__(self):
        self.api_key = None
        self.client = None
        self.tools_file = TOOLS_FILENAME
        
    def check_dependencies(self):
        """Check if required libraries are available."""
        try:
            import together
            import requests
            logger.info("All required dependencies are available")
            return True
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            print(f"Missing dependency: {e}")
            print("Please install required packages: pip install together requests")
            return False
    
    def setup_api_key(self):
        """Setup Together API key from environment or user input."""
        self.api_key = os.environ.get('TOGETHER_API_KEY')
        if not self.api_key:
            self.api_key = input("TOGETHER_API_KEY not detected. Please enter your Together API key: ")
            os.environ['TOGETHER_API_KEY'] = self.api_key
            logger.info("API key set in environment")
        
        try:
            from together import Together
            self.client = Together(api_key=self.api_key)
            logger.info("Together client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Together client: {e}")
            print(f"Failed to initialize Together client: {e}")
            return False
    
    def call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Make LLM call using Together API with DeepSeek R1."""
        try:
            response = self.client.chat.completions.create(
                model=TOGETHER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2048,
                temperature=0.1
            )
            output = response.choices[0].message.content
            logger.info("LLM call successful")
            return output
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def decompose_goal(self, user_goal: str) -> List[str]:
        """Decompose high-level goal into generic steps."""
        system_prompt = """You are a helpful assistant that decomposes high-level goals into a simple, numbered list of the most generic, atomic, and technology-agnostic steps possible. Do not provide solutions, only the steps. Example: Goal: 'Scrape a website and analyze the sentiment of the text' → Output: 1. Fetch text content from a URL. 2. Perform sentiment analysis on a text string."""
        
        user_prompt = f"Decompose this goal into generic steps: {user_goal}"
        
        try:
            response = self.call_llm(system_prompt, user_prompt)
            logger.info(f"Goal decomposition response: {response}")
            
            # Parse numbered steps first
            steps = []
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Try numbered format (1. step)
                match = re.match(r'^\d+\.\s*(.+)$', line)
                if match:
                    steps.append(match.group(1).strip())
                    continue
                
                # Try bullet format (- step)
                match = re.match(r'^[-*]\s*(.+)$', line)
                if match:
                    steps.append(match.group(1).strip())
                    continue
                
                # If no pattern matches, treat as a step if it's not empty
                if line and not line.startswith('Goal:') and not line.startswith('Output:'):
                    steps.append(line)
            
            if not steps:
                raise ValueError("No steps could be parsed from LLM response")
            
            logger.info(f"Parsed {len(steps)} steps: {steps}")
            return steps
            
        except Exception as e:
            logger.error(f"Goal decomposition failed: {e}")
            raise
    
    def scan_tools_file(self) -> Dict[str, Dict[str, Any]]:
        """Scan tools file for existing functions using AST."""
        tools = {}
        
        if not os.path.exists(self.tools_file):
            logger.info("Tools file does not exist, will create it")
            return tools
        
        try:
            with open(self.tools_file, 'r') as f:
                content = f.read()
            
            if not content.strip():
                return tools
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function signature
                    args = []
                    for arg in node.args.args:
                        if arg.arg != 'self':
                            args.append(arg.arg)
                    
                    # Extract docstring
                    docstring = ast.get_docstring(node) or ""
                    
                    tools[node.name] = {
                        'args': args,
                        'docstring': docstring,
                        'source': ast.unparse(node)
                    }
            
            logger.info(f"Found {len(tools)} existing tools")
            return tools
            
        except Exception as e:
            logger.error(f"Error scanning tools file: {e}")
            return {}
    
    def classify_tool_existence(self, step: str, existing_tools: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Classify if a generic tool exists for a given step."""
        context = []
        for name, info in existing_tools.items():
            context.append(f"{name}({', '.join(info['args'])}): {info['docstring']}")
        
        context_str = "\n".join(context) if context else "No existing tools"
        
        system_prompt = f"""You are a tool architect. Classify if a generic tool for a given step exists based on existing functions: {context_str}. Respond ONLY with a valid JSON object: {{"classification": "EXISTS_AND_GENERIC" | "EXISTS_BUT_SPECIFIC" | "NOT_EXISTS", "reason": "string"}}."""
        
        user_prompt = f"Step: {step}. Existing tools: {context_str}. Is there a generic tool? Respond with JSON."
        
        try:
            response = self.call_llm(system_prompt, user_prompt)
            logger.info(f"Tool classification response: {response}")
            
            # Try to parse JSON response
            try:
                result = json.loads(response)
                if 'classification' in result and 'reason' in result:
                    return result
            except json.JSONDecodeError:
                pass
            
            # Fallback regex parsing
            classification_match = re.search(r'"classification"\s*:\s*"([^"]+)"', response)
            reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', response)
            
            if classification_match and reason_match:
                return {
                    "classification": classification_match.group(1),
                    "reason": reason_match.group(1)
                }
            
            # Default fallback
            return {
                "classification": "NOT_EXISTS",
                "reason": "Could not parse LLM response"
            }
            
        except Exception as e:
            logger.error(f"Tool classification failed: {e}")
            return {
                "classification": "NOT_EXISTS",
                "reason": f"Classification failed: {e}"
            }
    
    def generate_tool_and_test(self, step: str) -> Dict[str, str]:
        """Generate function and test code for a given step."""
        system_prompt = """You are a world-class Python developer. Generate a generic, reusable function and a test case for it. **FUNCTION REQUIREMENTS:** - Name: descriptive, snake_case. - Arguments: typed, with sensible defaults if applicable. - Docstring: clearly describes purpose, args, returns. - Body: atomic, robust, with error handling. - Must be fully self-contained. Only use standard library imports. **TEST CASE REQUIREMENTS:** - Generate a `test_[function_name]` function. - The test should validate the function's logic. Since the agent runs in a virtual environment, the test may include necessary external setups (e.g., network calls, file operations) if required for the tool to function correctly. However, prefer mocks or safe checks where possible to avoid unnecessary side effects. - The test must return `True` if the function behaves as expected, `False` otherwise. **RESPONSE FORMAT:** Respond STRICTLY with a valid JSON object: {"filename": "tools.py", "code": "def example_func(...): ...", "test_code": "def test_example_func(): ... return True/False;"} Use double quotes. Escape newlines in the code strings."""
        
        user_prompt = f"Generate a generic, atomic function for: '{step}'. Make it standalone."
        
        try:
            response = self.call_llm(system_prompt, user_prompt)
            logger.info(f"Tool generation response: {response}")
            
            # Try to parse JSON response
            try:
                result = json.loads(response)
                if all(key in result for key in ['filename', 'code', 'test_code']):
                    return result
            except json.JSONDecodeError:
                pass
            
            # Fallback regex parsing
            code_match = re.search(r'"code"\s*:\s*"([^"]+)"', response)
            test_code_match = re.search(r'"test_code"\s*:\s*"([^"]+)"', response)
            
            if code_match and test_code_match:
                return {
                    "filename": "tools.py",
                    "code": code_match.group(1).replace('\\n', '\n'),
                    "test_code": test_code_match.group(1).replace('\\n', '\n')
                }
            
            raise ValueError("Could not parse tool generation response")
            
        except Exception as e:
            logger.error(f"Tool generation failed: {e}")
            raise
    
    def add_tool_to_file(self, code: str, test_code: str) -> bool:
        """Add new tool and test code to tools file."""
        try:
            # Create tools file if it doesn't exist
            if not os.path.exists(self.tools_file):
                with open(self.tools_file, 'w') as f:
                    f.write("# Auto-generated tools\n\n")
            
            # Read existing content
            with open(self.tools_file, 'r') as f:
                content = f.read()
            
            # Add new code
            new_content = content + "\n\n" + code + "\n\n" + test_code
            
            # Write back
            with open(self.tools_file, 'w') as f:
                f.write(new_content)
            
            logger.info("Tool and test code added to file")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add tool to file: {e}")
            return False
    
    def remove_tool_from_file(self, code: str, test_code: str) -> bool:
        """Remove tool and test code from tools file using AST."""
        try:
            with open(self.tools_file, 'r') as f:
                content = f.read()
            
            # Remove the added code
            lines = content.split('\n')
            code_lines = code.split('\n')
            test_lines = test_code.split('\n')
            
            # Find and remove the added code
            start_idx = None
            end_idx = None
            
            for i in range(len(lines)):
                if lines[i:i+len(code_lines)] == code_lines:
                    start_idx = i
                    end_idx = i + len(code_lines) + len(test_lines) + 2  # +2 for extra newlines
                    break
            
            if start_idx is not None:
                new_lines = lines[:start_idx] + lines[end_idx:]
                new_content = '\n'.join(new_lines)
                
                with open(self.tools_file, 'w') as f:
                    f.write(new_content)
                
                logger.info("Tool and test code removed from file")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove tool from file: {e}")
            return False
    
    def test_tool(self, test_function_name: str) -> bool:
        """Test a tool by running its test function."""
        try:
            # Import tools module
            if self.tools_file in sys.modules:
                importlib.reload(sys.modules[self.tools_file])
            else:
                importlib.import_module(self.tools_file.replace('.py', ''))
            
            # Get the test function
            tools_module = sys.modules[self.tools_file.replace('.py', '')]
            test_func = getattr(tools_module, test_function_name)
            
            # Run the test
            logger.info(f"Running test: {test_function_name}")
            result = test_func()
            
            if result is True:
                logger.info(f"Test {test_function_name} passed")
                return True
            else:
                logger.warning(f"Test {test_function_name} returned {result}")
                return False
                
        except Exception as e:
            logger.error(f"Test {test_function_name} failed with exception: {e}")
            return False
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool using subprocess."""
        try:
            cmd = [
                sys.executable, 
                AGENT_FILENAME, 
                "--execute-tool", 
                tool_name, 
                json.dumps(args)
            ]
            
            logger.info(f"Executing tool: {tool_name} with args: {args}")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            if result.returncode == 0:
                try:
                    output = json.loads(result.stdout.strip())
                    logger.info(f"Tool {tool_name} executed successfully")
                    return output
                except json.JSONDecodeError:
                    return {"output": result.stdout.strip(), "status": "success"}
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                logger.error(f"Tool {tool_name} executed successfully")
                return {"error": error_msg, "status": "failed"}
                
        except subprocess.TimeoutExpired:
            error_msg = f"Tool {tool_name} execution timed out"
            logger.error(error_msg)
            return {"error": error_msg, "status": "timeout"}
        except Exception as e:
            error_msg = f"Tool {tool_name} execution error: {e}"
            logger.error(error_msg)
            return {"error": error_msg, "status": "failed"}
    
    def get_tool_arguments(self, tool_name: str) -> Dict[str, Any]:
        """Get required arguments for a tool by parsing its signature."""
        try:
            tools_module = sys.modules[self.tools_file.replace('.py', '')]
            tool_func = getattr(tools_module, tool_name)
            
            sig = inspect.signature(tool_func)
            args = {}
            
            for param_name, param in sig.parameters.items():
                if param.default == inspect.Parameter.empty:
                    # Required parameter
                    if param_name != 'self':
                        user_input = input(f"Enter value for required parameter '{param_name}': ")
                        args[param_name] = user_input
                else:
                    # Optional parameter with default
                    args[param_name] = param.default
            
            return args
            
        except Exception as e:
            logger.error(f"Failed to get tool arguments: {e}")
            return {}
    
    def run_goal_execution(self, user_goal: str):
        """Main goal execution flow."""
        try:
            print(f"\n🎯 Goal: {user_goal}")
            print("=" * 50)
            
            # Phase 1: Goal decomposition
            print("\n📋 Phase 1: Decomposing goal into steps...")
            steps = self.decompose_goal(user_goal)
            print(f"Identified {len(steps)} steps:")
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step}")
            
            # Phase 2: Tool classification and creation
            print("\n🔧 Phase 2: Analyzing and creating tools...")
            existing_tools = self.scan_tools_file()
            
            execution_plan = []
            
            for i, step in enumerate(steps):
                print(f"\n  Step {i+1}: {step}")
                
                # Classify tool existence
                classification = self.classify_tool_existence(step, existing_tools)
                print(f"    Classification: {classification['classification']}")
                print(f"    Reason: {classification['reason']}")
                
                if classification['classification'] == 'EXISTS_AND_GENERIC':
                    # Find the existing tool
                    tool_name = None
                    for name, info in existing_tools.items():
                        if any(keyword in step.lower() for keyword in name.lower().split('_')):
                            tool_name = name
                            break
                    
                    if tool_name:
                        execution_plan.append({
                            'step': step,
                            'tool_name': tool_name,
                            'classification': classification['classification'],
                            'test_result': 'EXISTING_TOOL',
                            'execution_output': None
                        })
                        print(f"    Using existing tool: {tool_name}")
                    else:
                        classification['classification'] = 'NOT_EXISTS'
                        classification['reason'] = 'No matching existing tool found'
                
                if classification['classification'] in ['EXISTS_BUT_SPECIFIC', 'NOT_EXISTS']:
                    # Generate new tool
                    print(f"    Generating new tool...")
                    try:
                        tool_data = self.generate_tool_and_test(step)
                        
                        # Add tool to file
                        if self.add_tool_to_file(tool_data['code'], tool_data['test_code']):
                            # Extract function name from code
                            func_match = re.search(r'def\s+(\w+)\s*\(', tool_data['code'])
                            if func_match:
                                func_name = func_match.group(1)
                                test_func_name = f"test_{func_name}"
                                
                                # Test the tool
                                print(f"    Testing tool: {func_name}")
                                test_result = self.test_tool(test_func_name)
                                
                                if test_result:
                                    execution_plan.append({
                                        'step': step,
                                        'tool_name': func_name,
                                        'classification': 'NEW_TOOL_CREATED',
                                        'test_result': 'PASSED',
                                        'execution_output': None
                                    })
                                    print(f"    Tool {func_name} created and tested successfully")
                                    
                                    # Update existing tools
                                    existing_tools[func_name] = {
                                        'args': [],
                                        'docstring': '',
                                        'source': tool_data['code']
                                    }
                                else:
                                    # Remove failed tool
                                    self.remove_tool_from_file(tool_data['code'], tool_data['test_data'])
                                    raise Exception(f"Tool {func_name} executed successfully")
                            else:
                                raise Exception("Could not extract function name from generated code")
                        else:
                            raise Exception("Failed to add tool to file")
                            
                    except Exception as e:
                        logger.error(f"Tool creation failed for step {i+1}: {e}")
                        print(f"    ❌ Tool creation failed: {e}")
                        return
            
            # Phase 3: Goal execution
            print("\n🚀 Phase 3: Executing tools...")
            
            for i, plan_item in enumerate(execution_plan):
                print(f"\n  Executing step {i+1}: {plan_item['step']}")
                print(f"    Tool: {plan_item['tool_name']}")
                
                try:
                    # Get tool arguments
                    args = self.get_tool_arguments(plan_item['tool_name'])
                    
                    # Execute tool
                    output = self.execute_tool(plan_item['tool_name'], args)
                    plan_item['execution_output'] = output
                    
                    if output.get('status') == 'success':
                        print(f"    ✅ Success: {output.get('output', 'Tool executed successfully')}")
                    else:
                        print(f"    ❌ Failed: {output.get('error', 'Unknown error')}")
                        break
                        
                except Exception as e:
                    error_msg = f"Execution failed: {e}"
                    plan_item['execution_output'] = {'error': error_msg, 'status': 'error'}
                    print(f"    ❌ {error_msg}")
                    break
            
            # Phase 4: Reporting
            print("\n📊 Phase 4: Execution Report")
            print("=" * 50)
            
            for i, plan_item in enumerate(execution_plan):
                print(f"\nStep {i+1}: {plan_item['step']}")
                print(f"  Tool: {plan_item['tool_name']}")
                print(f"  Classification: {plan_item['classification']}")
                print(f"  Test Result: {plan_item['test_result']}")
                
                if plan_item['execution_output']:
                    if plan_item['execution_output'].get('status') == 'success':
                        print(f"  Execution: ✅ Success")
                        print(f"  Output: {plan_item['execution_output'].get('output', 'N/A')}")
                    else:
                        print(f"  Execution: ❌ Failed")
                        print(f"  Error: {plan_item['execution_output'].get('error', 'N/A')}")
                else:
                    print(f"  Execution: Not executed")
            
            print("\n" + "=" * 50)
            print("🎉 This agent has self-improved. Re-run it with a similar goal to see the existing tools detected and used instantly.")
            
        except Exception as e:
            logger.error(f"Goal execution failed: {e}")
            print(f"❌ Goal execution failed: {e}")
    
    def execute_tool_mode(self, tool_name: str, args_json: str):
        """Execute a tool in tool execution mode."""
        try:
            # Parse arguments
            args = json.loads(args_json)
            
            # Import tools module
            if self.tools_file in sys.modules:
                importlib.reload(sys.modules[self.tools_file])
            else:
                importlib.import_module(self.tools_file.replace('.py', ''))
            
            # Get the tool function
            tools_module = sys.modules[self.tools_file.replace('.py', '')]
            tool_func = getattr(tools_module, tool_name)
            
            # Execute the tool
            result = tool_func(**args)
            
            # Return result as JSON
            output = {"output": result, "status": "success"}
            print(json.dumps(output))
            
        except Exception as e:
            error_output = {"error": str(e), "status": "error"}
            print(json.dumps(error_output))
            sys.exit(1)

def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--execute-tool":
        if len(sys.argv) != 4:
            print(json.dumps({"error": "Invalid arguments for tool execution", "status": "error"}))
            sys.exit(1)
        
        tool_name = sys.argv[2]
        args_json = sys.argv[3]
        
        agent = Agent()
        agent.execute_tool_mode(tool_name, args_json)
        return
    
    # Normal agent mode
    agent = Agent()
    
    # Check dependencies
    if not agent.check_dependencies():
        sys.exit(1)
    
    # Setup API key
    if not agent.setup_api_key():
        sys.exit(1)
    
    # Get user goal
    user_goal = input("Please describe your high-level goal: ")
    
    # Run goal execution
    agent.run_goal_execution(user_goal)

if __name__ == "__main__":
    main()