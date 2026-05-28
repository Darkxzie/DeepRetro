# Chemistry Chatbot

A minimal Streamlit chatbot for chemistry and retrosynthesis questions. The app uses Streamlit's chat UI and OpenRouter's OpenAI-compatible API, so you can bring your own LLM token without storing secrets in the repository.

## What It Does

- Runs a browser-based chatbot at `http://localhost:8501`.
- Streams assistant responses from OpenRouter.
- Keeps chat history in the current Streamlit session.
- Provides a dark, Claude-inspired interface with a compact session sidebar.
- Uses a chemistry-focused system prompt for practical retrosynthesis and chemistry software help.
- Reads credentials from environment variables or Streamlit secrets.

## Files

- `streamlit_chatbot.py` - Streamlit UI and chat loop.
- `deepretro/utils/streamlit_chatbot.py` - settings, OpenRouter headers, and message helpers.
- `deepretro/tests/test_streamlit_chatbot.py` - focused tests for configuration and message handling.
- `env.example` - sample environment variables.

## Requirements

- Python 3.9+
- Streamlit
- OpenAI Python SDK
- An OpenRouter API key

Install the chatbot extra from the repo root:

```powershell
pip install -e .[chatbot]
```

If the OpenAI SDK is not already installed:

```powershell
pip install openai
```

## Configure

Set your OpenRouter key as an environment variable:

```powershell
$env:OPENROUTER_API_KEY='your-openrouter-key'
```

Optional settings:

```powershell
$env:OPENROUTER_MODEL='deepseek/deepseek-v4-flash:free'
$env:OPENROUTER_BASE_URL='https://openrouter.ai/api/v1'
$env:OPENROUTER_APP_NAME='Chemistry Chatbot'
```

You can also use Streamlit secrets at `.streamlit/secrets.toml`:

```toml
OPENROUTER_API_KEY = "your-openrouter-key"
OPENROUTER_MODEL = "deepseek/deepseek-v4-flash:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_APP_NAME = "Chemistry Chatbot"
```

`.streamlit/secrets.toml` is ignored by git. Do not commit real API keys.

## Run

```powershell
streamlit run streamlit_chatbot.py
```

Open:

```text
http://localhost:8501
```

Use the sidebar to choose a model, inspect the current provider settings, and start a new chat session.

## Testing

Run the focused chatbot tests:

```powershell
python -m pytest deepretro/tests/test_streamlit_chatbot.py -q
```

Compile-check the touched Python files:

```powershell
python -m compileall deepretro/utils/streamlit_chatbot.py streamlit_chatbot.py
```

## Notes

- The default model is `deepseek/deepseek-v4-flash:free`.
- OpenRouter free models can change over time. If a model is unavailable, choose another `:free` model from OpenRouter and set `OPENROUTER_MODEL`.
- The app is intentionally lightweight. It does not persist chats after the Streamlit session ends.
