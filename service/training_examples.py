"""可提交到训练接口的最小分类示例，无远程权重依赖。"""

from schemas.training import LOSSES, OPTIMIZERS, SCHEDULERS, TASK_TYPES, TrainingCreate


SOURCE = '''import torch
from torch.utils.data import Dataset

class Records(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        payload = self.records[index]["payload"]
        return {"inputs": {"x": torch.tensor(payload["features"], dtype=torch.float32)},
                "targets": torch.tensor(payload["label"], dtype=torch.int64)}

class Classifier(torch.nn.Module):
    def __init__(self, input_size, classes):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, classes)

    def forward(self, x):
        return self.linear(x)

def build_model(model_params):
    return Classifier(model_params.get("input_size", 2), model_params.get("classes", 2))

def build_dataset(records, data_params):
    return Records(records)
'''

CUSTOM_SOURCE = SOURCE + '''
def compute_loss(outputs, targets, loss_params):
    return torch.nn.functional.cross_entropy(outputs, targets, **loss_params)

def compute_metrics(validation_batches, metric_params):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    outputs = torch.cat([batch["outputs"] for batch in validation_batches])
    targets = torch.cat([batch["targets"] for batch in validation_batches])
    prediction = outputs.argmax(dim=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets.numpy(), prediction.numpy(), labels=list(range(outputs.shape[1])),
        average=metric_params.get("average", "macro"), zero_division=0)
    return {"accuracy": float(accuracy_score(targets.numpy(), prediction.numpy())),
            "precision": float(precision), "recall": float(recall), "f1": float(f1)}
'''


def example_request(custom=False):
    return {"name": "自定义分类示例" if custom else "分类训练示例", "dataset_id": 1,
            "version_id": "v1.0.0", "python_source": CUSTOM_SOURCE if custom else SOURCE,
            "contract": {"inputs": {"x": {"dtype": "float32", "shape": [None, 2]}},
                         "targets": {"dtype": "int64", "shape": [None]},
                         "outputs": {"dtype": "float32", "shape": [None, 2]}},
            "model_params": {"input_size": 2, "classes": 2},
            "loss": "custom" if custom else "auto", "metrics": "custom" if custom else "auto"}


def defaults():
    effective = TrainingCreate.model_validate(example_request()).model_dump()
    required = {"name", "dataset_id", "version_id", "python_source", "contract"}
    return {"defaults": {key: value for key, value in effective.items() if key not in required},
            "supported": {"task_types": TASK_TYPES, "optimizers": OPTIMIZERS,
                          "schedulers": SCHEDULERS, "losses": ["auto", *sorted(set(LOSSES.values()))],
                          "devices": ["auto", "cpu", "cuda"], "metrics": ["auto", "custom", "none"],
                          "averages": ["auto", "binary", "macro", "micro", "weighted", "samples"]},
            "example_record": {"id": "1", "payload": {"features": [0.2, 0.8], "label": 1}},
            "example_request": example_request(), "custom_example_request": example_request(True),
            "trusted_code_only": True}
