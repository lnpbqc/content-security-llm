"""只读查询业务 MySQL 中的数据集与原始记录，并提供治理规则。"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from db.governance_config import GovernanceConfigRepository



DIMENSIONS = ["文化价值", "信息价值", "稀缺性", "可信度", "代表性"]
ANOMALY_RULE_NAMES = [
    "主题标签一致性", "精确重复", "日期格式", "来源必填", "样本ID必填", "正文必填",
    "语种必填", "语种枚举", "标签枚举", "字段类型", "字符编码", "空白字符",
    "日期有效性", "元数据结构", "正文长度", "来源格式", "修订引用完整性",
    "重复组主记录", "必填字段非空", "字段白名单",
]
RISK_CATEGORIES = ["个人信息暴露", "误导信息", "仇恨歧视", "违法有害"]
RISK_LEVELS = [
    {"level": "HIGH", "label": "高风险", "color": "#f34d69", "definition": "有直接严重伤害或敏感个人信息暴露证据"},
    {"level": "MEDIUM", "label": "中风险", "color": "#f5a623", "definition": "存在可关联个人信息或明确的潜在伤害"},
    {"level": "LOW", "label": "低风险", "color": "#1687ff", "definition": "存在有限风险线索"},
    {"level": "NOTICE", "label": "提示", "color": "#8b5cf6", "definition": "需要补充核查的线索"},
]
RISK_RULES = [
    {"id": "PII-03", "version": "1.0", "category": "个人信息暴露", "name": "可关联身份信息核验",
     "text": "账户标识及注册时间等可关联信息需核验公开范围与授权依据。",
     "conditions": "账户标识与可关联元信息同时存在", "source": "演示风险分级方案 v1.0"},
    {"id": "MIS-01", "version": "1.0", "category": "误导信息", "name": "事实陈述来源核验",
     "text": "缺少可核验来源的新闻事实陈述需提示核查。",
     "conditions": "具体事件与时间但缺少可核验出处", "source": "演示风险分级方案 v1.0"},
]


def _connect_mysql(**kwargs):
    """延迟加载 MySQL 驱动，应用建单和启动时均不连接业务库。"""
    import pymysql

    return pymysql.connect(cursorclass=pymysql.cursors.DictCursor, **kwargs)


class BusinessGovernanceData:
    """从业务库读取数据，规则暂沿用本服务的固定治理方案。"""

    def __init__(self, config: GovernanceConfigRepository, cipher: Any,
                 connection_factory: Optional[Callable[..., Any]] = None,
                 mock_samples: bool = False):
        """保存 SQLite 配置入口；测试可注入模拟数据库连接。"""
        self.config = config
        self.cipher = cipher
        self.connection_factory = connection_factory or _connect_mysql
        self.mock_samples = mock_samples
        self._names = {}

    def _connect(self):
        """读取并解密管理员配置，连接失败时隐藏凭据。"""
        details = self.config.get_business_database()
        if details is None:
            raise RuntimeError("业务数据库连接未配置")
        try:
            return self.connection_factory(
                host=details["host"], port=details["port"], user=details["username"],
                password=self.cipher.decrypt(details["encrypted_password"].encode("ascii")).decode("utf-8"),
                database=details["database_name"], charset="utf8mb4",
                connect_timeout=10, read_timeout=30,
            )
        except Exception as exc:
            raise RuntimeError("业务数据库连接失败") from exc

    def policy(self, kind: str, scheme_id: str) -> Dict[str, Any]:
        """按任务类型和方案版本返回固定评分标准或规则。"""
        if kind == "governance-value" and scheme_id in {"general-v1", "general-v2"}:
            return {
                "scheme_id": scheme_id,
                "name": "通用价值评价 v1.0" if scheme_id == "general-v1" else "通用价值评价 v2.0",
                "dimensions": [{"name": name, "weight": 0.2,
                                "criteria": "仅依据样本文本评价{}；证据不足则标记不可评估。".format(name),
                                "anchors": "0 表示缺少相关信息，100 表示有充分文本证据。"}
                               for name in DIMENSIONS],
                "score_min": 0, "score_max": 100,
                "high_threshold": 85 if scheme_id == "general-v1" else 90,
                "medium_threshold": 60,
            }
        if kind == "governance-anomaly" and scheme_id == "anomaly-basic-v1":
            return {
                "scheme_id": scheme_id, "name": "基础异常检测 v1.0",
                "rule_version": scheme_id,
                "rules": [{"id": "anomaly-rule-{}".format(i + 1), "name": name,
                           "description": "按版本规则核查{}。".format(name)}
                          for i, name in enumerate(ANOMALY_RULE_NAMES)],
                "field_whitelist": ["topic_label"],
                "types": ["标签异常", "重复记录", "格式异常", "字段缺失"],
                "model_applicable_rule_ids": ["anomaly-rule-1"],
            }
        if kind == "governance-risk" and scheme_id == "risk-v1":
            return {"scheme_id": scheme_id, "name": "内容语义风险分级 v1.0",
                    "categories": RISK_CATEGORIES, "levels": RISK_LEVELS, "rules": RISK_RULES}
        raise ValueError("不支持的治理方案")

    def samples(self, scope: Dict[str, Any]) -> List[Dict[str, Any]]:
        """核对当前版本，并按原始记录 ID 顺序读取该数据集的全部样本。"""
        dataset_id = scope["dataset_id"]
        version_id = scope["version_id"]
        connection = self._connect()
        try:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT name, version FROM datasets WHERE id = %s", (dataset_id,))
                    dataset = cursor.fetchone()
                    if dataset is None:
                        raise ValueError("数据集不存在")
                    if dataset["version"] != version_id:
                        raise ValueError("数据集版本不匹配")
                    cursor.execute(
                        "SELECT id, payload FROM dataset_records WHERE dataset_id = %s ORDER BY id",
                        (dataset_id,),
                    )
                    rows = cursor.fetchall()
            except ValueError:
                raise
            except Exception as exc:
                raise RuntimeError("业务数据库查询失败") from exc
        finally:
            connection.close()
        if not rows:
            if not self.mock_samples:
                raise ValueError("数据集没有可治理的记录")
            # TODO: 业务后端完成文件解析入库并通过真实联调后，移除此临时分支。
            logging.getLogger(__name__).warning(
                "数据集 %s 版本 %s 无真实记录，使用临时模拟数据，非上传文件内容",
                dataset_id, version_id,
            )
            rows = self._mock_rows(dataset_id, version_id)
        samples = []
        for row in rows:
            payload = row["payload"]
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise ValueError("业务数据库记录 {} 的 payload 无效".format(row["id"])) from exc
            if not isinstance(payload, dict):
                raise ValueError("业务数据库记录 {} 的 payload 无效".format(row["id"]))
            sample_id = str(row["id"])
            language = payload.get("language") or payload.get("lang")
            samples.append({
                "id": sample_id, "dataset_id": dataset_id, "version_id": version_id,
                "revision_id": "{}:{}".format(version_id, sample_id),
                "text": self._record_text(payload),
                "language": language if isinstance(language, str) and language else "unknown",
                "metadata": payload,
            })
        self._names[(dataset_id, version_id)] = dataset["name"]
        return samples

    def _mock_rows(self, dataset_id: int, version_id: str) -> List[Dict[str, Any]]:
        """仅为尚未入库的数据集提供有明确来源标记的固定联调记录。"""
        examples = [
            ("跨文化交流", "交流活动中，参与者介绍各自的节日习俗，并讨论如何尊重文化差异。", "文化"),
            ("社区节庆", "社区通过节庆活动记录地方习俗与文化传承。", "体育"),
            ("虚构账户信息", "虚构演示记录：用户ID：10086；注册时间：2026/9/20。", "综合"),
            ("节日习俗交流", "同学分别介绍春节与开斋节的家庭活动，并共同讨论节日中的团聚与分享。", "文化"),
            ("地方技艺访谈", "口述访谈记录手工技艺的学习过程与文化传承，保留不同代际的表达方式。", "体育"),
            ("跨语言礼貌表达", "翻译练习比较不同语言中的问候与请求表达，说明使用场景和礼貌程度。", "教育"),
            ("校园饮食交流", "校园食堂邀请学生分享家乡饮食习惯，并通过清晰的食材标注尊重个人选择。", "文化"),
            ("基础问答", "问题：2 加 3 等于多少？回答：2 加 3 等于 5。", "教育"),
            ("虚构账户展示", "虚构演示记录：用户ID：20002；注册时间：2026/9/21；账户字段拟用于公开展示。", "综合"),
            ("虚构消息核验", "虚构演示消息：有人声称某地交流活动已取消，但没有提供主办方公告或可核验出处。", "新闻"),
        ]
        return [{"id": "mock-{}-{}-{}".format(dataset_id, version_id, index),
                 "payload": {"title": title, "text": text, "topic_label": label,
                             "language": "zh", "is_mock": True,
                             "source": "临时模拟数据（非上传文件内容）"}}
                for index, (title, text, label) in enumerate(examples, 1)]

    def _record_text(self, payload: Dict[str, Any]) -> str:
        """按字段名升序拼接全部原始字段，嵌套值使用稳定 JSON。"""
        lines = []
        for key, value in sorted(payload.items()):
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            lines.append("{}: {}".format(key, value))
        return "\n".join(lines)

    def dataset_name(self, dataset_id: int, version_id: str) -> str:
        """返回本次业务库查询取得的数据集名称。"""
        return self._names[(dataset_id, version_id)]

    def risk_text(self, text: str) -> str:
        """沿用现有用户 ID 遮蔽规则，供风险模型与结果页面使用。"""
        return re.sub(r"(用户ID：)\d+", r"\1[标识已遮蔽]", text)
