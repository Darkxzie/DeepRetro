from deepretro.utils.streamlit_chatbot import (
    ChatbotSettings,
    build_openrouter_messages,
    resolve_chatbot_settings,
)


def test_resolve_chatbot_settings_prefers_environment_values() -> None:
    settings = resolve_chatbot_settings(
        environ={
            "OPENROUTER_API_KEY": "env-key",
            "OPENROUTER_MODEL": "custom/model",
            "OPENROUTER_BASE_URL": "https://example.test/api",
            "OPENROUTER_APP_NAME": "DeepRetro Test",
        },
        secrets={"OPENROUTER_API_KEY": "secret-key"},
    )

    assert settings == ChatbotSettings(
        api_key="env-key",
        model="custom/model",
        base_url="https://example.test/api",
        app_name="DeepRetro Test",
        site_url=None,
    )


def test_resolve_chatbot_settings_strips_environment_whitespace() -> None:
    settings = resolve_chatbot_settings(
        environ={
            "OPENROUTER_API_KEY": " env-key\n",
            "OPENROUTER_MODEL": " deepseek/deepseek-v4-flash:free ",
        },
        secrets={},
    )

    assert settings.api_key == "env-key"
    assert settings.model == "deepseek/deepseek-v4-flash:free"


def test_resolve_chatbot_settings_falls_back_to_streamlit_secrets() -> None:
    settings = resolve_chatbot_settings(
        environ={},
        secrets={
            "OPENROUTER_API_KEY": "secret-key",
            "OPENROUTER_MODEL": "openrouter/free",
        },
    )

    assert settings.api_key == "secret-key"
    assert settings.model == "openrouter/free"
    assert settings.base_url == "https://openrouter.ai/api/v1"


def test_resolve_chatbot_settings_uses_concrete_free_default_model() -> None:
    settings = resolve_chatbot_settings(environ={}, secrets={})

    assert settings.model == "deepseek/deepseek-v4-flash:free"


def test_resolve_chatbot_settings_ignores_unreadable_streamlit_secrets() -> None:
    class MissingSecrets:
        def get(self, name: str) -> str:
            raise FileNotFoundError("No secrets files found")

    settings = resolve_chatbot_settings(
        environ={"OPENROUTER_API_KEY": "env-key"},
        secrets=MissingSecrets(),
    )

    assert settings.api_key == "env-key"
    assert settings.model == "deepseek/deepseek-v4-flash:free"


def test_build_openrouter_messages_keeps_supported_chat_roles() -> None:
    messages = build_openrouter_messages(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "ignored", "content": "Nope"},
        ],
        system_prompt="You are concise.",
    )

    assert messages == [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
