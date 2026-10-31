"""Regression checks for value evidence constraints and input isolation."""
import json
from types import SimpleNamespace
import pytest
from llm.governance_outputs import ValueSampleOutput
from llm.models import ModelManager
from service.governance import GovernanceService
from service.governance_data import BusinessGovernanceData

def test_value_evidence_required_in_provider_schema():
    schema=ValueSampleOutput.model_json_schema()
    assert schema["$defs"]["ValueDimensionOutput"]["properties"]["evidence"]["minItems"] == 1

def test_value_input_uses_body_without_source_metadata():
    service=GovernanceService(None,None,None,None)
    sample={"id":"2","text":"_source: internal\ntext: original body", "language":"en", "metadata":{"text":"original body","_source":{"file":"private"}}}
    policy=BusinessGovernanceData(None, connection_factory=lambda: None).policy("governance-value","general-v1")
    result=service._model_input("governance-value",sample,policy)
    assert result["text"] == "original body"
    assert "_source" not in json.dumps(result)

def test_error_call_preserves_diagnostics_without_changing_status():
    captured={}
    class Calls:
        def create(self,**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(id="call-test")
    manager=ModelManager.__new__(ModelManager)
    manager.calls=Calls()
    manager._log_error("model","token","input","governance_value_sample","invalid_structured_output",diagnostics={"raw_response":"{}","validation_error":"missing evidence"})
    assert captured["status"] == "error"
    assert json.loads(captured["output_json"])["validation_error"] == "missing evidence"
