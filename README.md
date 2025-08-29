# canvas-llm

Example implementation of a canvas LLM

The CLI tool allows users to first generate the initial canvas content based on their instructions, and then iteratively edit the content by specifying line numbers and new content. The CLI then shows the updated canvas content with the changes highlighted.

![Example usage of the canvas-llm tool](https://raw.githubusercontent.com/jakic12/canvas-llm/refs/heads/master/screenshot.png)

# Usage

First create an `.env` file with your OpenAI API key

```bash
echo "OPENAI_API_KEY=<your_api_key>" > .env
```

To run the main app

```bash
uv sync
```

```bash
python main.py
```

To run the performance tests

```bash
python testing.py
```
