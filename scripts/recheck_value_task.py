"""Reproduce selected samples or create a new value-analysis regression task."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Settings
from db.database import Database
from db.governance import GovernanceRepository
from db.governance_config import GovernanceConfigRepository
from llm.models import ModelManager, InvocationError
from service.governance import GovernanceService, OUTPUT_TYPES, RESPONSE_TYPES
from service.governance_data import BusinessGovernanceData
parser=argparse.ArgumentParser()
parser.add_argument("task_id")
parser.add_argument("--run-all", action="store_true")
args=parser.parse_args()
s=Settings.load(); db=Database(s); repo=GovernanceRepository(db)
with db.connect() as c:
    old=c.execute("SELECT * FROM llm_governance_tasks WHERE id=%s",(args.task_id,)).fetchone()
if not old or old["kind"]!="governance-value":
    raise SystemExit("Value task not found")
scope=json.loads(old["scope_json"])
data=BusinessGovernanceData(db); manager=ModelManager(db,s.credential_key)
service=GovernanceService(repo,data,manager,GovernanceConfigRepository(db))
if args.run_all:
    task=repo.create_task(old["kind"],old["token_hash"],"价值分析修复后回归",scope,old["model_id"],old["model_version"])
    with db.connect() as c:
        c.execute("UPDATE llm_governance_tasks SET status='running' WHERE id=%s",(task["id"],))
    print("NEW_TASK_ID="+task["id"],flush=True)
    service._run(task)
    with db.connect() as c:
        rows=c.execute("SELECT status,error_code,COUNT(*) n FROM llm_governance_task_samples WHERE task_id=%s GROUP BY status,error_code",(task["id"],)).fetchall()
    print(json.dumps(rows,ensure_ascii=True),flush=True)
else:
    policy=data.policy(old["kind"],scope["scheme_id"])
    for sample in data.samples(scope):
        if sample["id"] not in ("2","34391"): continue
        try:
            manager.invoke(old["model_id"],json.dumps(service._model_input(old["kind"],sample,policy),ensure_ascii=False),OUTPUT_TYPES[old["kind"]],response_type=RESPONSE_TYPES[old["kind"]],token_hash=old["token_hash"],validator=lambda v:service._validate(old["kind"],sample,policy,v.model_dump(mode="json")),timeout=90,max_retries=0)
            print(json.dumps({"sample":sample["id"],"status":"success"}),flush=True)
        except InvocationError as exc:
            print(json.dumps({"sample":sample["id"],"error":exc.code,"validation_error":exc.details.get("validation_error")},ensure_ascii=True),flush=True)
