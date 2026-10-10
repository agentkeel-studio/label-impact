"""label-impact: regulatory impact on one label, from the label table and one tool.

An agent from agentkeel's template (agentkeel-studio/agent-template at
66ca2e4, made from agentkeel 36c97dd). refagent's tool, schema and prompt
were written for a film distributor's rights table; this file rewrites the
tool for the label table and keeps everything else refagent's: Sonnet 4.6
through the pinned inference profile, one tool, no knowledge base, no HITL
branch and no memory.

Two things this file is careful about.

**The label table is the truth** (pharmadelta SPEC/00 §6, §9).
`find_label_row` returns one row and the CFR clauses it can be read under.
It does not decide whether the label is impacted or what is owed; the
model does, from the row and the clause.

**Where the row is read from.** The deployed runtime reads the DynamoDB
table the platform creates and loads from `data/table.json`
(`agentkeel-label-impact-rights`, agentkeel SPEC/06 §6 item 7). A run
outside the runtime reads `data/table.json` beside this file. Every
observation records which, in `source`.

The row's fields are pharmadelta `milestones/M00/fields.md`'s; the clause
ids are keys of `data/clauses.json`. The runtime image does not carry
`data/` (agentkeel `infra/construct/agent.Dockerfile`), so the clause ids a
section is read with are named here, as refagent names its own.

This module judges nothing and writes no envelope.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

BUNDLE = Path(__file__).parent
PROMPT = (BUNDLE / "prompt.txt").read_text(encoding="utf-8")
CONTRACT = json.loads((BUNDLE / "tools" / "find_label_row.json").read_text(encoding="utf-8"))
MANIFEST_PATH = BUNDLE / "manifest.yaml"
TABLE_FILE = BUNDLE / "data" / "table.json"

INFERENCE_CONFIG = {"temperature": 0, "maxTokens": 1024}
MAX_TOOL_CALLS = 2  # one call, and one more if the model corrects its arguments

# Which clauses a row can be read under, by its section. The row's own
# `requirement` comes first; these follow. The tool hands the candidates over;
# the question decides which one, and that is the model's to say (Tool Owner,
# the contract's `clause_candidates`). Every id is a key of data/clauses.json.
BY_SECTION = {
    "8.1": ["201.57(c)(9)(i)", "314.150(b)(10)"],
    "bw": ["314.150(b)(10)", "314.70(c)(6)(iii)(A)", "314.94(a)(8)(iv)", "314.97(a)"],
    "5": ["314.150(b)(10)", "314.70(c)(6)(iii)(A)", "314.94(a)(8)(iv)", "314.97(a)"],
    "5.1": ["314.150(b)(10)", "314.70(c)(6)(iii)(A)", "314.94(a)(8)(iv)", "314.97(a)"],
    "6": ["314.98(a)", "314.80(c)(1)(i)", "314.80(a)-serious", "314.80(a)-unexpected"],
}
NO_ROW = ["314.150(b)(10)"]  # what a question about a section we hold no row for is read under


def rights_rows(table_name: str | None, client: Any | None = None) -> tuple[list[dict[str, Any]], str]:
    """Every row, and where it was read. DynamoDB when there is a table, else the file.

    Named as refagent names it: `server.py` calls it by this name and is
    the template's, unchanged."""
    if table_name and client is not None:
        rows = []
        pages = client.get_paginator("scan").paginate(TableName=table_name)
        for page in pages:
            rows += [{k: _plain(v) for k, v in item.items()} for item in page["Items"]]
        return rows, "dynamodb"
    return json.loads(TABLE_FILE.read_text(encoding="utf-8")), "data/table.json"


def _plain(value: dict[str, Any]) -> Any:
    """One DynamoDB attribute, as the JSON file holds it."""
    if "S" in value:
        return value["S"]
    if "BOOL" in value:
        return value["BOOL"]
    if "NULL" in value:
        return None
    raise ValueError(f"unexpected attribute {value!r}")


def find_label_row(arguments: dict[str, Any], rows: list[dict[str, Any]], source: str) -> dict[str, Any]:
    """The tool. One row, and the clauses it can be read under. It decides nothing."""
    _refuse_unless_valid(arguments, CONTRACT["input"], "arguments")
    key = f"{arguments['label']}.{arguments['section']}"
    match = [row for row in rows if row["table_row"] == key]
    if len(match) > 1:
        # The table is keyed by table_row, unique. Two rows on one key means
        # the table does not say which governs, and choosing quietly would
        # make that the tool's decision.
        raise ValueError(f"find_label_row: {len(match)} rows for {key}")
    if not match:
        result = {"found": False, "row": None, "clause_candidates": list(NO_ROW), "source": source}
    else:
        row = match[0]
        candidates = [row["requirement"]]
        candidates += [c for c in BY_SECTION.get(row["section"], []) if c not in candidates]
        result = {"found": True, "row": row, "clause_candidates": candidates, "source": source}
    _refuse_unless_valid(result, CONTRACT["output"], "result")
    return result


def _refuse_unless_valid(value: Any, schema: dict[str, Any], what: str) -> None:
    """Strict, both ways (Tool Owner). A tool that accepts what its schema forbids has no schema."""
    from jsonschema import Draft202012Validator

    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=str)
    if errors:
        raise ValueError(f"find_label_row {what}: " + "; ".join(e.message for e in errors))


def tool_config() -> dict[str, Any]:
    return {"tools": [{"toolSpec": {
        "name": CONTRACT["name"],
        "description": CONTRACT["description"],
        "inputSchema": {"json": CONTRACT["input"]},
    }}]}  # fmt: skip


def parse_json(text: str) -> dict[str, Any] | None:
    """The outermost JSON object in the reply, or None. No repair."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def guardrail_config(guardrail: dict[str, str] | None) -> dict[str, Any]:
    """converse's guardrailConfig for the manifest's pin, or nothing when there is none."""
    if not guardrail:
        return {}
    return {"guardrailConfig": {"guardrailIdentifier": guardrail["id"], "guardrailVersion": guardrail["version"],
                                "trace": "enabled"}}  # fmt: skip


def question_content(question: str, guardrail: dict[str, str] | None) -> list[dict[str, Any]]:
    """The user's turn. Under a guardrail, the question is the only input it assesses (guardContent).

    The system prompt and the tool's results (the table's rows) are not the
    user's words. The model reads guardContent as it reads text. The answer
    is assessed whole.
    """
    return [{"guardContent": {"text": {"text": question}}}] if guardrail else [{"text": question}]


def intervening_topics(trace: dict[str, Any]) -> list[str]:
    """Every denied topic the guardrail's trace says it blocked on, input and output."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for topic in node.get("topicPolicy", {}).get("topics", []):
                if topic.get("action") == "BLOCKED":
                    found.add(topic["name"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(trace.get("guardrail", {}))
    return sorted(found)


def answer(client: Any, question: str, model_id: str, rows: list[dict[str, Any]], source: str,
           guardrail: dict[str, str] | None = None) -> dict[str, Any]:  # fmt: skip
    """Converse, with the tool, until the model answers. Returns the raw observation; judges nothing.

    `guardrail` is the manifest's pin, `{"id", "version"}`, or None: the
    runner passes it from the manifest, the runtime from the environment
    the platform sets from the same manifest.
    """
    started = time.perf_counter()
    messages: list[dict[str, Any]] = [{"role": "user", "content": question_content(question, guardrail)}]
    topics: set[str] = set()
    usage = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
    calls: list[dict[str, Any]] = []
    stop_reason = ""

    failed = None
    for _ in range(MAX_TOOL_CALLS + 1):
        try:
            response = client.converse(
                modelId=model_id,
                system=[{"text": PROMPT}],
                messages=messages,
                inferenceConfig=INFERENCE_CONFIG,
                toolConfig=tool_config(),
                **guardrail_config(guardrail),
            )
        except (BotoCoreError, ClientError) as exc:
            # A golden that spends and then fails has still spent; the usage
            # of the calls before it is kept.
            failed = f"{type(exc).__name__}: {exc}"
            break
        for field in usage:
            usage[field] += response["usage"].get(field, 0)
        stop_reason = response["stopReason"]
        topics.update(intervening_topics(response.get("trace", {})))
        reply = response["output"]["message"]
        messages.append(reply)
        uses = [block["toolUse"] for block in reply["content"] if "toolUse" in block]
        if not uses:
            break
        results = []
        for use in uses:
            try:
                result: Any = find_label_row(use["input"], rows, source)
                status = "success"
            except ValueError as exc:  # the model's arguments, or our own result: it is told either way
                result, status = {"error": str(exc)}, "error"
            calls.append({"name": use["name"], "input": use["input"], "status": status, "output": result})
            results.append({"toolResult": {"toolUseId": use["toolUseId"],
                                           "content": [{"json": result}], "status": status}})  # fmt: skip
        messages.append({"role": "user", "content": results})

    text = "".join(block.get("text", "") for block in messages[-1]["content"] if isinstance(block, dict))
    observation = {"error": failed} if failed else {}
    return {
        **observation,
        "text": text,
        "parsed": parse_json(text),
        "stop_reason": stop_reason,
        "guardrail_topics": sorted(topics),
        "usage": usage,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "tool_calls": calls,
        "source": source,
    }
