from tools.validate_deepseek_v2_lite_assets import validate_chat_template


CHAT_TEMPLATE = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"""


def test_chat_template_contract_accepts_pinned_shape():
    assert validate_chat_template({"chat_template": CHAT_TEMPLATE}) == []


def test_chat_template_contract_rejects_missing_template():
    assert validate_chat_template({}) == [
        "tokenizer_config.json has no nonempty chat_template"
    ]


def test_chat_template_contract_requires_assistant_termination():
    errors = validate_chat_template(
        {"chat_template": CHAT_TEMPLATE.replace("eos_token", "end_marker")}
    )
    assert errors == ["chat_template is missing required fragment 'eos_token'"]


def test_chat_template_contract_rejects_invalid_jinja():
    errors = validate_chat_template({"chat_template": CHAT_TEMPLATE + "{% endif %}"})
    assert len(errors) == 1
    assert errors[0].startswith("chat_template cannot be rendered:")
