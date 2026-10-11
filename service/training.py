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

    def logs(self, task_id, token_hash, after_offset=0, limit=65536):
        task = self.get(task_id, token_hash)
        filename = "run.log" if task["config"].get("runner") == "script" else "stderr.log"
        root = self.repository.directory(task_id).resolve()
        path = (root / filename).resolve()
        if not path.is_relative_to(root):
            raise HTTPException(404, detail={"code": "log_unavailable", "message": "日志不可用"})
        if not path.is_file():
            return {"text": "", "next_after_offset": after_offset, "has_more": False}
        size = path.stat().st_size
        if after_offset > size:
            raise HTTPException(422, detail={"code": "invalid_log_offset", "message": "日志游标超出文件大小"})
        with path.open("rb") as stream:
            if after_offset:
                stream.seek(after_offset - 1)
                if stream.read(1) != b"\n" and after_offset != size:
                    raise HTTPException(422, detail={"code": "invalid_log_offset", "message": "请使用接口返回的完整行游标"})
            stream.seek(after_offset)
            chunk = stream.read(limit)
        terminal = task["status"] in ("succeeded", "failed", "canceled")
        if not (terminal and after_offset + len(chunk) == size):
            end = chunk.rfind(b"\n") + 1
            if not end and len(chunk) == limit:
                raise HTTPException(422, detail={"code": "log_line_too_long", "message": "请增大 limit 以容纳完整日志行"})
            chunk = chunk[:end]
        offset = after_offset + len(chunk)
        return {"text": chunk.decode("utf-8", errors="replace"), "next_after_offset": offset,
                "has_more": offset < size}
