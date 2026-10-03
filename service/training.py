"""训练建单、只读结果与鉴权产物定位；HTTP 请求不加载 PyTorch 或执行源码。"""

from fastapi import HTTPException


class TrainingService:
    def __init__(self, repository):
        self.repository = repository

    def get(self, task_id, token_hash):
        task = self.repository.get(task_id, token_hash)
        if task is None:
            raise HTTPException(404, detail={"code": "training_task_not_found", "message": "训练任务不存在"})
        return task

    def artifact_path(self, task_id, artifact_id, token_hash):
        self.get(task_id, token_hash)
        artifact = next((item for item in self.repository.artifacts(task_id) if item["id"] == artifact_id), None)
        if artifact is None:
            raise HTTPException(404, detail={"code": "artifact_not_found", "message": "产物不存在"})
        root = self.repository.directory(task_id).resolve()
        path = (root / artifact["filename"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(404, detail={"code": "artifact_unavailable", "message": "产物文件不可用"})
        return path
