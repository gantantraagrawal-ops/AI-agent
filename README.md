# 🤖 Self-Improving Goal Execution Agent

A production-grade, autonomous AI agent that creates, tests, and executes tools to achieve high-level goals while continuously improving its capabilities.

## 🎯 Core Features

- **Self-Verifying**: Creates tools and immediately tests them with comprehensive validation
- **Goal-Executing**: Decomposes complex goals into atomic steps and executes them sequentially
- **Self-Improving**: Saves successfully tested tools to a library for future reuse
- **Stateless**: No memory between runs - improvement is proven when re-running finds existing tools
- **Ultra-Robust**: Handles all failures gracefully with comprehensive error handling

## 🚀 Quick Start

### Prerequisites

1. **Install required packages:**
```bash
pip install together requests
```

2. **Get a Together API key:**
   - Visit [Together AI](https://together.ai/)
   - Sign up and obtain your API key
   - The agent will prompt you for it if not set in environment

### Run the Agent

```bash
python agent.py
```

The agent will:
1. Check dependencies
2. Prompt for API key if needed
3. Ask for your high-level goal
4. Automatically decompose, create tools, test them, and execute your goal

## 🏗️ Architecture

### Execution Flow

1. **Goal Decomposition** → Uses DeepSeek R1 to break goals into atomic steps
2. **Tool Analysis** → Scans existing tools and classifies what's needed
3. **Tool Creation** → Generates new tools with comprehensive test cases
4. **Tool Testing** → Immediately validates new tools (including external setups)
5. **Goal Execution** → Runs tools sequentially to achieve the goal
6. **Tool Library** → Saves successful tools for future use

### Key Components

- **Agent Class**: Main orchestrator for goal execution
- **LLM Integration**: Together API with DeepSeek R1 model
- **AST-based Code Management**: Safe code parsing and manipulation
- **Subprocess Security**: Isolated tool execution for safety
- **Comprehensive Logging**: Both console and file logging

## 📁 File Structure

```
├── agent.py          # Main agent implementation
├── tools.py          # Auto-generated tools library (created by agent)
├── agent.log         # Detailed execution logs
└── README.md         # This documentation
```

## 🔧 Usage Examples

### Basic Goal Execution
```bash
$ python agent.py
TOGETHER_API_KEY not detected. Please enter your Together API key: your_key_here
Please describe your high-level goal: Scrape a website and analyze the sentiment of the text

🎯 Goal: Scrape a website and analyze the sentiment of the text
==================================================
📋 Phase 1: Decomposing goal into steps...
Identified 2 steps:
  1. Fetch text content from a URL
  2. Perform sentiment analysis on a text string
...
```

### Tool Execution Mode
```bash
python agent.py --execute-tool tool_name '{"arg1": "value1"}'
```

## 🛡️ Security Features

- **No eval()/exec()**: Uses subprocess for safe tool execution
- **Isolated Execution**: Tools run in separate processes
- **Input Validation**: Comprehensive argument validation
- **Error Boundaries**: Graceful failure handling

## 🔍 Tool Management

### Tool Classification
The agent classifies tools into three categories:
- **EXISTS_AND_GENERIC**: Perfect match, use immediately
- **EXISTS_BUT_SPECIFIC**: Similar tool exists but needs adaptation
- **NOT_EXISTS**: No suitable tool, create new one

### Tool Testing
- **Immediate Validation**: Tools are tested right after creation
- **External Setup Support**: Tests can include network calls, file operations, etc.
- **Failure Handling**: Failed tools are automatically removed
- **Comprehensive Coverage**: Tests validate both logic and functionality

## 📊 Logging and Monitoring

- **Console Output**: Real-time progress and status updates
- **File Logging**: Detailed logs saved to `agent.log`
- **Log Levels**: INFO, WARNING, ERROR with timestamps
- **Execution Tracking**: Complete audit trail of all operations

## 🚨 Error Handling

The agent handles various failure scenarios:
- **Missing Dependencies**: Graceful exit with installation instructions
- **API Failures**: Retry logic and fallback mechanisms
- **Tool Creation Failures**: Automatic rollback and cleanup
- **Execution Timeouts**: Configurable timeout handling
- **External Failures**: Robust error reporting and recovery

## 🔧 Configuration

### Environment Variables
```bash
export TOGETHER_API_KEY="your_api_key_here"
```

### LLM Configuration
- **Model**: DeepSeek R1 via Together AI
- **Max Tokens**: 2048
- **Temperature**: 0.1 (for consistent outputs)
- **Timeout**: 30 seconds per tool execution

## 📈 Self-Improvement Evidence

The agent demonstrates self-improvement through:
1. **Tool Reuse**: Subsequent runs find and use existing tools instantly
2. **Library Growth**: Tools file accumulates tested, working functions
3. **Efficiency Gains**: Faster execution on similar goals
4. **Quality Improvement**: Only successful tools are retained

## 🧪 Testing Philosophy

- **Test-First Development**: Tools are only created with comprehensive tests
- **External Setup Support**: Tests can perform necessary external operations
- **Failure Tolerance**: Tests handle external failures gracefully
- **Validation Focus**: Tests must return `True` for success

## 🔮 Future Enhancements

- **Tool Versioning**: Track tool evolution and improvements
- **Performance Metrics**: Measure execution time and success rates
- **Tool Composition**: Combine existing tools for complex operations
- **Learning Integration**: Incorporate execution feedback for tool improvement

## 🤝 Contributing

This agent is designed to be completely autonomous, but contributions to the core architecture are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install together requests
   ```

2. **API Key Issues**
   - Verify your Together AI API key is valid
   - Check account has sufficient credits

3. **Permission Errors**
   - Ensure write permissions in current directory
   - Check Python environment permissions

4. **Network Issues**
   - Verify internet connectivity
   - Check firewall settings for API calls

### Getting Help

- Check the `agent.log` file for detailed error information
- Review console output for specific error messages
- Ensure all dependencies are properly installed
- Verify API key configuration

---

**🎉 Ready to experience autonomous AI tool creation and goal execution!**

Run `python agent.py` and describe your goal - the agent will handle the rest autonomously.