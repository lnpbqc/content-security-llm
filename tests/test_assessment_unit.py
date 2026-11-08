"""Availability, repair and fallback regressions; real SDK, simulated provider, no DB."""
import json
import unittest
from datetime import datetime
from unittest.mock import patch

from llm import analysis_contracts as a, governance_outputs as g, outputs as o
from llm.assessment import UNAVAILABLE, unavailable_result
from llm.models import InvocationError
from llm.structured_output import StructuredOutputValidationError
from db.entities import CallRecord
from schemas.calls import CallPublic
from service.analysis import validate_audit, validate_scenario
from tests import test_structured_output as harness


class AssessmentTests(unittest.TestCase):
    setUp = harness.StructuredOutputTests.setUp
    invoke = harness.StructuredOutputTests.invoke

    def test_invalid_json_is_repaired_once_and_both_attempts_are_recorded(self):
        self.content = ['broken JSON', '{"answer":"修正后的回答"}']
        result = self.invoke()
        self.assertEqual(result.answer, "修正后的回答")
        self.assertEqual(result.assessment_status, "assessed")
        self.assertEqual(len(self.requests), 2)
        self.assertIn('待修正数据', self.requests[1]['messages'][-1]['content'])
        self.assertEqual([c.kwargs['status'] for c in self.calls.create.call_args_list], ['error', 'success'])

    def test_second_provider_failure_returns_unavailable(self):
        self.content, self.status = '{}', [200, 503]
        self.assertEqual(self.invoke().assessment_status, 'unavailable')
        self.assertEqual(len(self.requests), 2)

    def test_first_provider_failure_is_not_disguised_as_insufficient_materials(self):
        for status in (401, 429, 503):
            self.status = status
            before = len(self.requests)
            with self.assertRaises(InvocationError) as caught:
                self.invoke()
            self.assertEqual(caught.exception.code, 'provider_error')
            self.assertEqual(caught.exception.upstream_status, status)
            self.assertEqual(len(self.requests) - before, 1)
            self.assertNotIn('private', str(caught.exception))

    def test_repair_uses_remaining_budget_and_does_not_use_sdk_retries(self):
        self.content = ['{}', '{"answer":"fixed"}']
        options = []
        factory = self.manager.client_factory
        def tracked(**kwargs):
            client = factory(**kwargs)
            original = client.with_options
            def configure(**values):
                options.append(values)
                return original(**values)
            client.with_options = configure
            return client
        self.manager.client_factory = tracked
        with patch('llm.models.monotonic', side_effect=[0, 1, 45]):
            self.invoke(timeout=60)
        self.assertEqual(options, [dict(timeout=59, max_retries=0), dict(timeout=15, max_retries=0)])

    def test_exhausted_budget_skips_repair(self):
        self.content = '{}'
        with patch('llm.models.monotonic', side_effect=[0, 0, 61]):
            result = self.invoke(timeout=60)
        self.assertEqual(result.assessment_status, 'unavailable')
        self.assertEqual(len(self.requests), 1)

    def test_valid_unavailable_result_does_not_retry_or_expose_rejected_governance_text(self):
        self.content = json.dumps(dict(originalOutput='姓名：张小明', governedOutput='姓名：张小明',
                                       reason='资料不足', riskLevel='unknown'))
        result = self.invoke(o.ModelRiskGovernanceOutput)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(result.governedOutput, UNAVAILABLE)
        self.assertNotIn('张小明', result.model_dump_json())

    def test_real_zero_score_is_distinct_from_unavailable_zero(self):
        self.content = '{"score":0,"reason":"依据材料，实际为零"}'
        assessed = self.invoke(o.JudgementAnswer)
        self.content = '{}'
        unavailable = self.invoke(o.JudgementAnswer)
        self.assertEqual(assessed.score, unavailable.score)
        self.assertEqual(assessed.assessment_status, 'assessed')
        self.assertEqual(unavailable.assessment_status, 'unavailable')
        self.assertIn('score', unavailable.unavailable_fields)

    def test_partial_scenario_keeps_only_validated_evidence(self):
        request = dict(scenario_code='public_opinion', content='原文', model_id='model-test', materials=[])
        data = unavailable_result(a.PublicOpinionOutput, json.dumps(request)).model_dump()
        data.update(assessment_status='partial', evidence=[dict(source_id='input', quote='原文')])
        self.content = json.dumps(data)
        result = self.invoke(a.PublicOpinionOutput, input=json.dumps(request),
                             validator=lambda v: validate_scenario(v, a.ScenarioRequest(**request)))
        self.assertEqual(result.assessment_status, 'partial')
        self.assertEqual(result.evidence[0].quote, '原文')
        self.assertEqual(len(self.requests), 1)

    def test_fabricated_citation_is_not_accepted_as_partial(self):
        request = dict(scenario_code='public_opinion', content='原文', model_id='model-test', materials=[])
        data = unavailable_result(a.PublicOpinionOutput, json.dumps(request)).model_dump()
        data.update(assessment_status='partial', evidence=[dict(source_id='input', quote='伪造引用')])
        self.content = json.dumps(data)
        result = self.invoke(a.PublicOpinionOutput, input=json.dumps(request),
                             validator=lambda v: validate_scenario(v, a.ScenarioRequest(**request)))
        self.assertEqual(result.assessment_status, 'unavailable')
        self.assertEqual(result.evidence, [])
        self.assertEqual(len(self.requests), 2)

    def test_programming_errors_are_not_swallowed(self):
        def broken(_):
            raise RuntimeError('programming error')
        with self.assertRaisesRegex(RuntimeError, 'programming error'):
            self.invoke(validator=broken)

    def test_private_diagnostics_are_not_in_public_call_history(self):
        record = CallRecord('call-1', 'model-1', 'token', 'request',
                            json.dumps({'_diagnostics': {'raw_response': 'sensitive rejected output'}}),
                            'analysis_public_opinion', 'error', 'result_validation_failed', datetime.now())
        self.assertIsNone(CallPublic.from_record(record).output)

    def test_all_builtin_templates_are_typed_without_invented_findings(self):
        request = dict(sample_id='s1', sampleId='s1', sampleRevisionId='rev1', audit_type='full_chain',
                       subject=dict(id='subject', type='model', version='v1'),
                       steps=[dict(id='step1', rule_refs=['r1'])])
        for cls in list(o._response_types.values()) + [g.ValueSampleOutput, g.AnomalyLabelJudgment,
                g.AnomalySampleOutput, g.AnomalyRepairOutput, g.RiskSampleOutput, g.MultilingualSampleOutput]:
            with self.subTest(model=cls.__name__):
                result = unavailable_result(cls, json.dumps(request))
                self.assertEqual(result.assessment_status, 'unavailable')
                self.assertTrue(result.assessment_reason)
                self.assertFalse(getattr(result, 'findings', []))
                self.assertFalse(getattr(result, 'fieldChanges', []))

    def test_scenario_and_audit_templates_pass_domain_checks(self):
        classes = [a.PublicOpinionOutput, a.HotEventsOutput, a.CrossCulturalOutput,
                   a.EthnicGovernanceOutput, a.SmartGovernmentOutput, a.CyberSecurityOutput]
        for cls in classes:
            code = cls.model_fields['scenario_code'].annotation.__args__[0]
            request = a.ScenarioRequest(model_id='m', scenario_code=code, content='原文',
                                       target_language='en' if code == 'cross_cultural' else None)
            validate_scenario(unavailable_result(cls, request.model_dump_json()), request)
        for cls, kind in [(a.ReasoningAuditOutput, 'reasoning'), (a.ChainAuditOutput, 'full_chain')]:
            request = a.AuditRequest(model_id='m', audit_type=kind, subject=dict(id='m', type='model', version='v1'),
                                     steps=[dict(id='step1', label='step', detail='detail', rule_refs=['r1'])],
                                     rules=[dict(id='r1', text='rule')], evidence=[])
            result = unavailable_result(cls, request.model_dump_json())
            validate_audit(result, request)
            self.assertEqual(result.subject, request.subject)


if __name__ == '__main__':
    unittest.main()
