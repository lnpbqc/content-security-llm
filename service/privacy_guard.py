"""模型治理的隐私兜底及复检；只对明确格式和有上下文的个人信息作规则处理。

保留 originalOutput 作为审计原文。规则不代表对任意敏感信息的完整语义审核，
也不把“正文有变化”当作所有风险已经消除的证明。
"""

import re

from llm.outputs import ModelRiskGovernanceOutput
from llm.structured_output import StructuredOutputValidationError


_PATTERNS = (
    ("手机号", re.compile(r"(?<![0-9A-Za-z])(?:\+?86[ \t-]?)?1[3-9](?:[ \t-]?[0-9]){9}(?![0-9A-Za-z])")),
    ("身份证号", re.compile(
        r"(?<![0-9A-Za-z])[1-9][0-9]{5}(?:19|20)[0-9]{2}"
        r"(?:0[1-9]|1[0-2])(?:0[1-9]|[12][0-9]|3[01])[0-9]{3}[0-9Xx](?![0-9A-Za-z])"
    )),
    ("邮箱", re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")),
)
_NAME = re.compile(
    r"(?P<label>(?:(?:顾客|客户|联系人|收件人|用户)姓名|顾客|客户|联系人|收件人|真实姓名|姓名))"
    r"[ \t]*(?P<colon>[:：]?)[ \t]*"
    r"(?P<name>[\u3400-\u9fff]{2,6}(?:[·•][\u3400-\u9fff]{1,6})?)"
    r"(?=$|[\s，,。；;、！!？?：:])"
)
_SURNAMES = frozenset(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐"
    "费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平"
    "黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝"
    "董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊"
    "胡凌霍虞万支柯管卢莫经房裘缪干解应宗丁宣邓郁单杭洪包左石崔吉龚"
    "程嵇邢裴陆荣翁荀甄曲封储靳段富巫焦巴牧山谷车侯全班仰秋仲伊宁仇"
    "栾暴甘厉戎祖武符刘景詹龙叶幸司黎薄印宿白怀蒲台从鄂索赖卓蔺屠蒙"
    "池乔阴胥能苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍桑桂濮牛寿通边扈"
    "燕冀浦尚农温庄晏柴瞿阎连习艾鱼容向古易慎戈廖庾终居衡步都耿满弘"
)
_MASKED_NAMES = {"已脱敏", "已遮蔽", "已隐藏", "已匿名", "匿名", "脱敏", "隐藏", "记录", "信息"}


def _find_sensitive(text: str) -> dict[str, str]:
    found = {}
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            found[match.group()] = kind
    for match in _NAME.finditer(text):
        name = match.group("name")
        if name not in _MASKED_NAMES and (match.group("colon") or name[0] in _SURNAMES):
            found[name] = "姓名"
    return found


def _redact(text: str, findings: dict[str, str]) -> str:
    if not findings:
        return text
    # 一次替换，优先匹配较长原值，避免替换结果再次被当成待替换文本。
    kinds = {value.casefold(): kind for value, kind in findings.items()}
    pattern = re.compile("|".join(re.escape(value) for value in sorted(findings, key=len, reverse=True)), re.I)
    return pattern.sub(lambda match: f"[{kinds[match.group().casefold()]}已脱敏]", text)


def enforce_privacy(output: ModelRiskGovernanceOutput, source_text: str) -> None:
    """作为 ModelManager 的 validator，在成功记录保存前处理并复检结果。

    同时检查输入、原始输出及治理正文，以免模型误判 low、换格式或省略原文
    导致已知敏感原值漏过；错误消息不得带出具体的个人信息。
    """
    if any(not value.strip() for value in (output.originalOutput, output.governedOutput, output.reason)):
        raise StructuredOutputValidationError("治理正文及判定依据不能为空")

    original_findings = _find_sensitive(output.originalOutput)
    governed_findings = _find_sensitive(output.governedOutput)
    findings = _find_sensitive(source_text)
    for text in (output.originalOutput, output.governedOutput, output.reason, output.reconstruction or ""):
        findings.update(_find_sensitive(text))

    before = output.governedOutput
    output.governedOutput = _redact(output.governedOutput, findings)
    output.reason = _redact(output.reason, findings)
    if output.reconstruction is not None:
        output.reconstruction = _redact(output.reconstruction, findings)

    if original_findings or governed_findings or any(
        value.casefold() in (output.originalOutput + before).casefold() for value in findings
    ):
        output.riskLevel = "high"

    for text in (output.governedOutput, output.reason, output.reconstruction or ""):
        if _find_sensitive(text) or any(value.casefold() in text.casefold() for value in findings):
            raise StructuredOutputValidationError("隐私复检未通过，治理结果未放行，请复核")

    if output.riskLevel in {"high", "medium"} and output.originalOutput.strip() == output.governedOutput.strip():
        raise StructuredOutputValidationError("检测到风险但正文未完成治理，结果未放行，请复核")

    if findings:
        kinds = "、".join(sorted(set(findings.values())))
        action = "已补充脱敏并复检" if before != output.governedOutput else "已复检"
        output.reason += f"\n服务端隐私规则{action}（检查范围：{kinds}）。"
