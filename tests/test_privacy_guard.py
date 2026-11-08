"""治理结果隐私规则：既验证漏脱敏兜底，也验证正常内容及原始证据保持不变。"""

import pytest

from llm.outputs import ModelRiskGovernanceOutput
from llm.structured_output import StructuredOutputValidationError
from service.privacy_guard import enforce_privacy


ORIGINAL = "顾客张小明，手机号码：13800138000，刚刚下单了一份黄焖鸡米饭。"
REDACTED = "顾客[姓名已脱敏]，手机号码：[手机号已脱敏]，刚刚下单了一份黄焖鸡米饭。"


def outcome(original=ORIGINAL, governed=None, risk="high", **kwargs):
    return ModelRiskGovernanceOutput(
        originalOutput=original, governedOutput=original if governed is None else governed,
        riskLevel=risk, reason=kwargs.pop("reason", "包含个人信息，需要保护。"), **kwargs,
    )


@pytest.mark.parametrize("risk", ["high", "medium", "low"])
def test_unchanged_customer_record_is_redacted_even_if_model_misclassifies_it(risk):
    result = outcome(risk=risk)
    enforce_privacy(result, ORIGINAL)
    assert result.originalOutput == ORIGINAL
    assert result.governedOutput == REDACTED
    assert result.riskLevel == "high"
    assert "已补充脱敏并复检" in result.reason


def test_cosmetic_rewrite_still_requires_phone_and_name_redaction():
    result = outcome(governed=ORIGINAL.replace("刚刚", "刚才"))
    enforce_privacy(result, ORIGINAL)
    assert result.governedOutput == REDACTED.replace("刚刚", "刚才")


def test_known_name_from_input_is_removed_from_unlabelled_output_and_explanations():
    result = outcome(
        original="张小明刚刚下单了黄焖鸡米饭。", risk="low",
        reason="张小明的联系方式为13800138000。",
        reconstruction="整理张小明的订单。",
    )
    enforce_privacy(result, ORIGINAL)
    assert result.governedOutput == "[姓名已脱敏]刚刚下单了黄焖鸡米饭。"
    assert result.riskLevel == "high"
    for field in (result.governedOutput, result.reason, result.reconstruction):
        assert "张小明" not in field and "13800138000" not in field


@pytest.mark.parametrize(("text", "expected"), [
    ("联系电话：+86 138-0013-8000。", "联系电话：[手机号已脱敏]。"),
    ("手机号：138 0013 8000。", "手机号：[手机号已脱敏]。"),
    ("身份证号：110101199001011234。", "身份证号：[身份证号已脱敏]。"),
    ("邮箱：test.user@example.com。", "邮箱：[邮箱已脱敏]。"),
    ("联系人：李四，收件人：王小红。", "联系人：[姓名已脱敏]，收件人：[姓名已脱敏]。"),
    ("姓名：阿依古丽。", "姓名：[姓名已脱敏]。"),
])
def test_supported_formats_are_masked(text, expected):
    result = outcome(original=text, risk="low")
    enforce_privacy(result, text)
    assert result.governedOutput == expected


def test_model_redacted_result_is_preserved_and_original_risk_is_not_downgraded():
    result = outcome(governed=REDACTED)
    enforce_privacy(result, ORIGINAL)
    assert result.governedOutput == REDACTED and result.originalOutput == ORIGINAL
    assert "已复检" in result.reason and "已补充脱敏" not in result.reason
    assert result.riskLevel == "high"


@pytest.mark.parametrize("text", [
    "顾客今天下单，联系人记录已整理，姓名已遮蔽。",
    "客户服务正常，订单编号：202611060001。",
    "姓名：[姓名已脱敏]，手机号：[手机号已脱敏]。",
    "北京今天晴天，建议携带饮用水。",
])
def test_non_sensitive_text_and_existing_placeholders_are_not_corrupted(text):
    result = outcome(original=text, risk="low", reason="未发现风险。")
    enforce_privacy(result, text)
    assert result.governedOutput == text and result.reason == "未发现风险。"
    assert result.riskLevel == "low"


@pytest.mark.parametrize("risk", ["high", "medium"])
def test_unhandled_risk_is_rejected_instead_of_pretending_governance_succeeded(risk):
    result = outcome(original="尚未处理的风险正文。", risk=risk)
    with pytest.raises(StructuredOutputValidationError, match="正文未完成治理"):
        enforce_privacy(result, result.originalOutput)


@pytest.mark.parametrize("field", ["originalOutput", "governedOutput", "reason"])
def test_whitespace_results_are_rejected(field):
    result = outcome()
    setattr(result, field, " \n ")
    with pytest.raises(StructuredOutputValidationError, match="不能为空"):
        enforce_privacy(result, ORIGINAL)
