import pytest
from langgraph.checkpoint.memory import MemorySaver
from src.talk_to_data.graph import CreditRiskAgent


@pytest.fixture(scope="module")
def agent():
    return CreditRiskAgent(checkpointer=MemorySaver())


def test_gold_q1_overall_default_rate(agent):
    res = agent.ask("What is the overall default rate?", thread_id="test_gold")
    assert res.get("tool_used") == "Database Query"
    answer = res.get("final_answer", "")
    assert "8.07%" in answer or "8.1%" in answer or "0.0807" in answer


def test_gold_q2_highest_risk_age_band(agent):
    res = agent.ask("Which age band has the highest default rate?", thread_id="test_gold")
    assert res.get("tool_used") == "Database Query"
    answer = res.get("final_answer", "")
    assert "20-29" in answer
    assert "11.4%" in answer or "0.114" in answer or "11.44%" in answer


def test_gold_q3_compare_education_levels(agent):
    res = agent.ask("Compare default rates across education levels.", thread_id="test_gold")
    assert res.get("tool_used") == "Database Query"
    assert res.get("sql_valid") is True
    assert "education_summary" in res.get("sql", "").lower() or "name_education_type" in res.get("sql", "").lower()


def test_gold_q4_late_payment_behaviour(agent):
    res = agent.ask("How does previous late-payment behaviour relate to default?", thread_id="test_gold")
    assert res.get("tool_used") == "Database Query"
    answer = res.get("final_answer", "")
    assert res.get("sql_valid") is True


def test_gold_q5_ext_source_3_meaning(agent):
    res = agent.ask("What does EXT_SOURCE_3 mean?", thread_id="test_gold")
    assert res.get("tool_used") == "Knowledge Base"
    answer = res.get("final_answer", "").lower()
    assert "score" in answer or "external" in answer or "normalized" in answer


def test_gold_q6_why_ebm_selected(agent):
    res = agent.ask("Why was EBM selected?", thread_id="test_gold")
    assert res.get("tool_used") == "Knowledge Base"
    answer = res.get("final_answer", "").lower()
    assert "glass-box" in answer or "interpretability" in answer or "additive" in answer or "ebm" in answer


def test_gold_q7_probability_of_default_general(agent):
    res = agent.ask("What is probability of default in general? Use external sources.", thread_id="test_gold")
    assert res.get("tool_used") == "Web Search (DDGS)"
    answer = res.get("final_answer", "").lower()
    assert "probability of default" in answer or "borrower" in answer or "likelihood" in answer


def test_gold_q8_follow_up_uses_conversation_memory(agent):
    res = agent.ask("Can you summarize the findings from the previous question?", thread_id="test_gold")
    assert res.get("tool_used") == "Conversation Memory"
    assert res.get("conversation")
    answer = res.get("final_answer", "").lower()
    assert "probability of default" in answer or "previous response" in answer


def test_gold_q9_destructive_query_blocked(agent):
    res = agent.ask(
        "DROP TABLE analytics.applicants; (Security Injection Test)",
        thread_id="test_gold",
    )
    assert res.get("tool_used") == "SQL Safety Guard"
    answer = res.get("final_answer", "")
    assert "cannot execute that query safely" in answer
    assert "Drop" in answer or "Prohibited operation" in answer or "Only SELECT" in answer
