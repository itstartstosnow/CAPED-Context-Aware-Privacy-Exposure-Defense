"""Task content requirements extractor using the /api/v1/llm_2b endpoint.

设计
====
LLM 输出 {app, subject, type}，to_expanded() 方法将其直接包装为
ExpandedRequirement，不查 Table A，不做包名解析。
包名匹配推迟到 Phase 2 每步实时执行（privacy_classifier._find_requirement）。
具体元素显示策略由 effective context + canonical policy table 在每步实时查询。

LLM 输出格式：
[
  {"app": "微信",          "subject": "老王",        "type": "person"},
  {"app": "高德地图",      "subject": "all",          "type": "all"},
  {"app": "Simple Gallery","subject": "travel photo", "type": "item"}
]

字段说明：
  app     — 任务中提到的 App 名称（或推断的 App 类型）
  subject — "all"（该 App 所有内容）或 自然语言描述（VQA/文本匹配使用）
  type    — person / item / date / all / none

to_expanded() 逻辑（直通包装，无 DB 查询）：
  ExpandedRequirement(
    app_names    = [req.app]    # LLM 原文，逐字存储
    subject_type = req.type
    subject      = req.subject
  )
  无 packages 字段，无 screen_ctx。
  运行时 _find_requirement(expanded_reqs, current_pkg) 通过反查 DB 匹配包名。

缓存：相同任务文本不重复推理。
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import string
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote

import requests

from android_world.agents.privacy.adb_helper import run_adb_text

logger = logging.getLogger(__name__)

# ── LLM model endpoints ────────────────────────────────────────────────────────

# Maps model alias → server path.  Add new models here as needed.
LLM_ENDPOINTS: dict[str, str] = {
    "0.8b": "/api/v1/llm_08b",
    "2b":   "/api/v1/llm_2b",
}
DEFAULT_LLM_MODEL: str = "0.8b"
_PIXEL_APPSSEARCH_AUTHORITY = "com.google.android.apps.nexuslauncher.appssearch"
_PIXEL_QUERY_PREFIXES = tuple(string.ascii_lowercase)
_INSTALLED_APP_LABELS_CACHE = Path(__file__).with_name("installed_app_labels_cache.json")

# ── Type constants ──────────────────────────────────────────────────────────────

_VALID_TYPE_NAMES: frozenset[str] = frozenset(
    {"person", "item", "date", "all", "none"}
)

_GUIDED_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "app":     {"type": "string"},
            "subject": {"type": "string"},
            "type": {
                "type": "string",
                "enum": sorted(_VALID_TYPE_NAMES),  # includes "none"
            },
        },
        "required": ["app", "subject", "type"],
        "additionalProperties": False,
    },
}

# ── Prompt V2: Structured CoT (Think lines before JSON) ───────────────────────
# Each example has a "Think:" line (or numbered steps for multi-app) that forces
# explicit reasoning about app identity, subject semantics, and type — before
# committing to the JSON output.  The parser (_parse) is unaffected because it
# extracts the last [...] JSON array from the full generated text.
# Use extract_v2() / _call_llm_v2() to run this variant.

_SYSTEM_PROMPT_V2 = """\
You are completing a task on this phone.
You can use the apps installed on this phone to complete it.
To complete the task, you need to navigate through the installed apps, perform actions, and see specific content in those apps.

First, THINK: What apps will you need to use, and what action will you perform in each app? Combine similar operations on the same app into one step. Keep to at most 5 steps.

Then, OUTPUT JSON: For each piece of content you need to access, specify which app it's in, its exact name (or "all" for new/unspecified content), and its type.

Each JSON item: {"app": "<app name>", "subject": "<subject_name or 'all'>", "type": "person" | "item" | "date" | "all" | "none"}

subject: specific subject_name = the exact name of content you need to find/access
         "all" = all content, or new content that doesn't exist yet
         "none" = just navigate to app, no content needed

type: person = person's name or phone number
      item = named existing object
      date = date/time value
      all = new or general/unspecified content
      none = navigate only

Rules:
1. For any **write operation that creates or enters new content**, use subject="all", type="all". This includes create, add, write, compose, record, log, fill in, enter details, and import. Do NOT use subject_name with type="person"/"item"/"date" unless the task must first find existing content. Names, phone numbers, addresses, note text, amounts in write operations are payload to enter — not existing content to locate.
2. Infer implied apps when needed: photo/video→Camera; files→Files; timer/stopwatch→Clock; wifi/bluetooth/brightness→Settings. Every output "app" must be an exact app name from the installed apps list below. Do not invent app names outside the list.
3. One JSON entry per distinct (app, subject_name) pair. Do NOT repeat the same (app, subject_name).
4. Think steps are planning only — keep at most 5 steps, combine similar operations.

Installed apps on this phone:
{installed_app_list}

Example:
Task: Delete doc_a.pdf and doc_b.pdf from the Downloads folder in Files.
Think:
1. Files | find Downloads folder and find doc_a.pdf and doc_b.pdf, delete them
[{"app":"Files","subject":"Downloads","type":"item"},{"app":"Files","subject":"doc_a.pdf","type":"item"},{"app":"Files","subject":"doc_b.pdf","type":"item"}]

Example:
Task: Write a new note in Markor, write content, and save it.
Think:
1. Markor | create note, write content, save
[{"app":"Markor","subject":"all","type":"all"}]

Example:
Task: Create a new contact for PERSON_1. Their number is PHONE_1.
Think:
1. Contacts | create a new contact and enter the provided details
[{"app":"Contacts","subject":"all","type":"all"}]

Example:
Task: Add a new event to Calendar for dinner with PERSON_1 at 7pm.
Think:
1. Calendar | create a new event and enter the provided details
[{"app":"Calendar","subject":"all","type":"all"}]

Example:
Task: Open the existing contact card for PERSON_1 in Contacts.
Think:
1. Contacts | find the existing contact PERSON_1
[{"app":"Contacts","subject":"PERSON_1","type":"person"}]

Example:
Task: Find message from PERSON_1 in WeChat and navigate with Maps.
Think:
1. WeChat | find message from PERSON_1
2. Maps | navigate to location
[{"app":"WeChat","subject":"PERSON_1","type":"person"},{"app":"Maps","subject":"all","type":"all"}]

Example:
Task: Set a reminder to pick up PERSON_1 from the airport at 8pm.
Think:
1. Reminders | create new reminder (write op, PERSON_1 is reminder content)
[{"app":"Reminders","subject":"all","type":"all"}]

Example:
Task: Open the Weather app.
Think:
1. Weather | open
[{"app":"Weather","subject":"none","type":"none"}]

Example:
Task: Read the entries in data.csv from Files and import them into the Tracker app.
Think:
1. Files | find and read data.csv
2. Tracker | import new records
[{"app":"Files","subject":"data.csv","type":"item"},{"app":"Tracker","subject":"all","type":"all"}]

Example:
Task: Find the PDF attachment report.pdf that PERSON_1 sent me in Telegram, then open it with with PDF reader and review it.
Think:
1. Telegram | find message from PERSON_1 and open attachment report.pdf
2. PDF reader | open and review document
[{"app":"Telegram","subject":"PERSON_1","type":"person"},{"app":"Telegram","subject":"report.pdf","type":"item"},{"app":"PDF reader","subject":"all","type":"all"}]

Example:
Task: Send a text message to +14015478520 with a specific message via Simple SMS Messenger.
Think:
1. Simple SMS Messenger | send to phone number
[{"app":"Simple SMS Messenger","subject":"PHONE_1","type":"person"}]

Example:
Task: 看看 PERSON_3 最近给我发了什么短信
Think:
1. Messenger | find message from PERSON_3
[{"app":"Messenger","subject":"PERSON_3","type":"person"}]
"""

# ── Prompt V1 Retry (same examples as V2, but no Think lines, for fallback) ───
# Used when V2 parse fails (e.g. repetition loop exhausts token budget).
# guided_json constrains the entire output to be a valid JSON array, so
# Think lines cannot appear and _parse() succeeds reliably.
# No {installed_app_list} — kept short to fit within max_tokens=80.

_SYSTEM_PROMPT_V1_RETRY = """\
Given a task on a mobile phone, output a JSON array listing what content in which apps the agent needs to access.
Each item: {"app": "<app name>", "subject": "<subject_name or 'all' or 'none'>", "type": "person" | "item" | "date" | "all" | "none"}
Output JSON array only, no markdown or explanation.

subject: specific name = exact content to find; "all" = all/new content; "none" = just navigate
type: person=name/phone; item=named object; date=date/time; all=new or general; none=navigate only

Rules:
1. For any write operation (create, add, write, log, fill in, enter details), use subject="all", type="all".
2. Infer implied apps: photo/video→Camera; files→Files; timer/stopwatch→Clock; wifi/brightness→Settings.
3. One entry per distinct (app, subject) pair. Do NOT repeat the same (app, subject).

Example:
Task: Delete doc_a.pdf and doc_b.pdf from the Downloads folder in Files.
[{"app":"Files","subject":"Downloads","type":"item"},{"app":"Files","subject":"doc_a.pdf","type":"item"},{"app":"Files","subject":"doc_b.pdf","type":"item"}]

Example:
Task: Write a new note in Markor, write content, and save it.
[{"app":"Markor","subject":"all","type":"all"}]

Example:
Task: Create a new contact for PERSON_1. Their number is PHONE_1.
[{"app":"Contacts","subject":"all","type":"all"}]

Example:
Task: Add a new event to Calendar for dinner with PERSON_1 at 7pm.
[{"app":"Calendar","subject":"all","type":"all"}]

Example:
Task: Open the existing contact card for PERSON_1 in Contacts.
[{"app":"Contacts","subject":"PERSON_1","type":"person"}]

Example:
Task: Find message from PERSON_1 in WeChat and navigate with Maps.
[{"app":"WeChat","subject":"PERSON_1","type":"person"},{"app":"Maps","subject":"all","type":"all"}]

Example:
Task: Set a reminder to pick up PERSON_1 from the airport at 8pm.
[{"app":"Reminders","subject":"all","type":"all"}]

Example:
Task: Open the Weather app.
[{"app":"Weather","subject":"none","type":"none"}]

Example:
Task: Read the entries in data.csv from Files and import them into the Tracker app.
[{"app":"Files","subject":"data.csv","type":"item"},{"app":"Tracker","subject":"all","type":"all"}]

Example:
Task: Find the PDF attachment report.pdf that PERSON_1 sent me in Telegram, then open it with PDF reader and review it.
[{"app":"Telegram","subject":"PERSON_1","type":"person"},{"app":"Telegram","subject":"report.pdf","type":"item"},{"app":"PDF reader","subject":"all","type":"all"}]

Example:
Task: Send a text message to +14015478520 with a specific message via Simple SMS Messenger.
[{"app":"Simple SMS Messenger","subject":"PHONE_1","type":"person"}]

Example:
Task: 看看 PERSON_3 最近给我发了什么短信
[{"app":"Messenger","subject":"PERSON_3","type":"person"}]
"""

# ── Prompt V1 (original, direct JSON output) ──────────────────────────────────

_SYSTEM_PROMPT = """\
Given a task on a mobile phone, output a JSON array listing what content in which apps the agent needs to see.
Each item: {"app": "<app name>", "subject": "all" | "none" | <brief description>, "type": "person" | "item" | "date" | "all" | "none"}
Output JSON array only, no markdown or explanation.

subject: "all" = all content in that app; specific description = content to find via matching.
type:
  person   — subject is a person's name or phone number (contact, sender, recepient); not message text
  item     — subject is a named item (product name, note title, event, etc.)
  date     — subject is a date, time, or time range
  all      — need all content, or content is general/unspecified
  none     — app must be opened/navigated to, but no content inside needs to be seen

Rules:
1. Include one entry per app per distinct piece of content needed.
2. Use subject="all", type="all" when the specified content does not yet exist on the phone (creating a new note, write a new file, creating a new contact — the new item does not exist yet therefore nothing to match against). Use a specific subject + type only when the target already exists and must be located first (a person in a chat list, a named file, an item in a list).
3. If an app is implied but not named (take photo → Camera; manage files → Files; change settings → Settings, etc.), use the implied app name.
4. Even if you only need to navigate to one app without reading its content, it must be included in the list with type="none".
5. Important contrast: if the task is creating a new record or entering details into a new record, then names, phone numbers, addresses, note text, amounts, and similar values are payload to enter, not existing content to search for. In such cases, use subject="all", type="all". Use person/item/date only when the task must first locate content that already exists on the phone.

Task: Take one photo.
[{"app":"Camera","subject":"all","type":"all"}]
Task: Open the Weather app.
[{"app":"Weather","subject":"none","type":"none"}]
Task: Write a new note named project_ideas.txt in the Noto app.
[{"app":"Noto","subject":"all","type":"all"}]
Task: Delete the following items from Shopping List: Milk, Bread.
[{"app":"Shopping List","subject":"Milk","type":"item"},{"app":"Shopping List","subject":"Bread","type":"item"}]
Task: Log the following expenses to the Budget Tracker app: Lunch $12, Coffee $3. Then send the summary to PHONE_2 via WhatsApp.
[{"app":"Budget Tracker","subject":"all","type":"all"},{"app":"WhatsApp","subject":"PHONE_2","type":"person"}]
Task: Move the file holiday_photos.jpg from the Podcasts folder to the DCIM folder using Files.
[{"app":"Files","subject":"holiday_photos.jpg","type":"item"},{"app":"Files","subject":"Podcasts","type":"item"},{"app":"Files","subject":"DCIM","type":"item"}]
Task: Find the restaurant 老王 recommended in yesterday's 微信 message and navigate there with 高德地图
[{"app":"微信","subject":"老王","type":"person"},{"app":"高德地图","subject":"all","type":"all"}]
Task: Send an WhatsApp message to PHONE_3 with this message: The quick brown fox jumps over the lazy dog.
[{"app":"WhatsApp","subject":"PHONE_3","type":"person"}]
Task: Pick a travel photo from my album and post it to Instagram
[{"app":"Photos","subject":"travel photo","type":"item"},{"app":"Instagram","subject":"all","type":"all"}]
Task: 看看张三最近给我发了什么短信
[{"app":"Messenger","subject":"张三","type":"person"}]
Task: Add a new contact named Alice Bob, phone number +1-222-3333.
[{"app":"Contacts","subject":"all","type":"all"}]
Task: Add a new event to Calendar for dinner with PERSON_1 at 7pm.
[{"app":"Calendar","subject":"all","type":"all"}]
Task: Open the existing contact card for Alice Bob in Contacts.
[{"app":"Contacts","subject":"Alice Bob","type":"person"}]
Task: How many notes are in my Work folder in the Notes app?
[{"app":"Notes","subject":"all","type":"all"}]
"""


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclasses.dataclass
class AppRequirement:
    """A single content requirement extracted from the task (LLM output).

    app:     App display name as mentioned in (or inferred from) the task.
    subject: "all" to access all content, or natural-language description for matching.
    type:    One of {person, item, date, all}.
    """
    app: str
    subject: str
    type: str


@dataclasses.dataclass
class TaskRequirements:
    """Structured task requirements for the expand pipeline.

    requirements: list of AppRequirement extracted from the task.
    Empty list = no content needed → all content masked by default.
    """
    requirements: List[AppRequirement] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ExpandedRequirement:
    """A single requirement after app name resolution.

    app_names:    All registered display names for this app (e.g. ["WeChat", "微信"]).
                  Used by LauncherPolicy for launcher-screen icon matching.
    subject_type: One of {person, item, date, all}.
    subject:      Matching subject value ("all" or natural-language description).

    Note: Package matching is done at runtime via reverse DB lookup in
    _find_requirement(), not via pre-resolved package lists.
    """
    app_names: List[str]
    subject_type: str
    subject: str


@dataclasses.dataclass
class ExpandedRequirements:
    """Resolved requirements ready for privacy_classifier and LauncherPolicy.

    requirements: List of ExpandedRequirement (one per app per content item).
                  Phase 2 fast-paths PUBLIC context directly via level==PUBLIC check;
                  no separate auto_expose_apps set is needed.
    """
    requirements: List[ExpandedRequirement] = dataclasses.field(default_factory=list)


def _parse_launcher_components_from_query_activities(text: str) -> set[tuple[str, str]]:
    """Parse `cmd package query-activities --brief` output into components."""
    components: set[tuple[str, str]] = set()
    for line in text.splitlines():
        line = line.strip()
        if not re.match(r"^[A-Za-z0-9._$-]+/.*$", line):
            continue
        pkg, activity = line.split("/", 1)
        if pkg and activity:
            components.add((pkg, activity))
    return components


def _parse_pixel_search_rows(text: str) -> dict[tuple[str, str], str]:
    """Parse Pixel Launcher search rows into {(pkg, activity): label}."""
    labels: dict[tuple[str, str], str] = {}
    for line in text.splitlines():
        match = re.search(
            r"suggest_text_1=(.*?), .*?component=([^&]+)&",
            line,
        )
        if not match:
            continue
        label = match.group(1).strip()
        component = unquote(match.group(2))
        if "/" not in component or not label:
            continue
        pkg, activity = component.split("/", 1)
        if pkg and activity:
            labels[(pkg, activity)] = label
    return labels


def _query_pixel_launcher_labels(
    launcher_components: set[tuple[str, str]],
    adb_path: str = "adb",
    device_serial: Optional[str] = None,
    timeout: float = 10.0,
) -> dict[tuple[str, str], str]:
    """Query Pixel Launcher search provider for exact app labels."""
    labels: dict[tuple[str, str], str] = {}
    for prefix in _PIXEL_QUERY_PREFIXES:
        uri = f"content://{_PIXEL_APPSSEARCH_AUTHORITY}/search_suggest_query/{prefix}"
        try:
            out = run_adb_text(
                adb_path,
                device_serial,
                "shell",
                "content",
                "query",
                "--uri",
                uri,
                timeout=timeout,
            )
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                f"TaskExtractor: Pixel launcher label query failed for prefix {prefix!r}"
            ) from exc
        for component, label in _parse_pixel_search_rows(out).items():
            if component in launcher_components and component not in labels:
                labels[component] = label
    return labels


def _load_cached_installed_app_labels() -> list[str]:
    """Load expand/extend privacy task fallback app labels.

    This is a project-local patch for expand/extend privacy task pilot runs
    where Pixel Launcher's appssearch provider can be slow or unavailable. It
    does not change AndroidWorld official task registration or app discovery.
    """
    try:
        with _INSTALLED_APP_LABELS_CACHE.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "TaskExtractor: cached installed app labels unavailable"
        ) from exc

    labels = data.get("labels", [])
    if not isinstance(labels, list):
        raise RuntimeError("TaskExtractor: cached installed app labels malformed")

    cleaned = sorted(
        {str(label).strip() for label in labels if str(label).strip()},
        key=str.casefold,
    )
    if not cleaned:
        raise RuntimeError("TaskExtractor: cached installed app labels empty")
    return cleaned


def _get_installed_app_labels(
    adb_path: str = "adb",
    device_serial: Optional[str] = None,
    timeout: float = 10.0,
) -> list[str]:
    """Return exact launcher-visible app labels via Pixel Launcher provider."""
    try:
        query_output = run_adb_text(
            adb_path,
            device_serial,
            "shell",
            "cmd",
            "package",
            "query-activities",
            "--brief",
            "-a",
            "android.intent.action.MAIN",
            "-c",
            "android.intent.category.LAUNCHER",
            timeout=timeout,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "TaskExtractor: failed to list launcher activities; using cached "
            "expand/extend task app labels fallback. This patch is project-local "
            "and does not affect AndroidWorld official task registration. Error: %s",
            exc,
        )
        return _load_cached_installed_app_labels()

    launcher_components = _parse_launcher_components_from_query_activities(query_output)
    if not launcher_components:
        logger.warning(
            "TaskExtractor: no launcher-visible apps found from adb query; using "
            "cached expand/extend task app labels fallback. This patch is "
            "project-local and does not affect AndroidWorld official task registration."
        )
        return _load_cached_installed_app_labels()

    try:
        labels_by_component = _query_pixel_launcher_labels(
            launcher_components,
            adb_path=adb_path,
            device_serial=device_serial,
            timeout=timeout,
        )
    except RuntimeError as exc:
        logger.warning(
            "TaskExtractor: Pixel Launcher app label query failed; using cached "
            "expand/extend task app labels fallback. This patch is project-local "
            "and does not affect AndroidWorld official task registration. Error: %s",
            exc,
        )
        return _load_cached_installed_app_labels()

    if not labels_by_component:
        logger.warning(
            "TaskExtractor: Pixel Launcher query returned no exact app labels; "
            "using cached expand/extend task app labels fallback. This patch is "
            "project-local and does not affect AndroidWorld official task registration."
        )
        return _load_cached_installed_app_labels()

    return sorted(set(labels_by_component.values()), key=str.casefold)


# ── App DB loader (shared with screen_context — single in-memory cache) ────────
from android_world.agents.privacy.launcher_policy import _norm_app_name as _norm  # noqa: E402


# ── TaskExtractor ─────────────────────────────────────────────────────────────

class TaskExtractor:
    """Extract structured task requirements from a natural-language task string.

    API:
        extract(task) -> TaskRequirements    (LLM → {app, subject, type})
        expand(reqs)  -> ExpandedRequirements ({app_names, pkg, subject_type, subject})

    Args:
        server_base_url: Base URL of the agentPrivacy server.
        timeout: HTTP request timeout in seconds.
        llm_model: LLM model alias ("0.8b" or "2b"). Controls which server
                   endpoint is used for task extraction inference.
                   See LLM_ENDPOINTS for the full mapping.
    """

    def __init__(
        self,
        server_base_url: str = "http://localhost:8001",
        timeout: float = 30.0,
        llm_model: str = DEFAULT_LLM_MODEL,
        adb_path: str = "adb",
        device_serial: Optional[str] = None,
    ):
        self._base_url = server_base_url.rstrip("/")
        self._timeout = timeout
        self._adb_path = adb_path
        self._device_serial = device_serial
        self._session = requests.Session()
        self._cache: Dict[str, TaskRequirements] = {}
        self._llm_path = LLM_ENDPOINTS.get(llm_model, LLM_ENDPOINTS[DEFAULT_LLM_MODEL])
        logger.info("TaskExtractor: llm_model=%r → endpoint=%s", llm_model, self._llm_path)

    # ── Public API ─────────────────────────────────────────────────────────────

    def to_expanded(self, reqs: TaskRequirements) -> ExpandedRequirements:
        """Convert TaskRequirements to ExpandedRequirements.

        App names come directly from TaskExtractor output.
        Requirement matching is done at runtime via reverse lookup in _find_requirement().
        """
        expanded: List[ExpandedRequirement] = []

        for req in reqs.requirements:
            expanded.append(ExpandedRequirement(
                app_names=[req.app],
                subject_type=req.type,
                subject=req.subject,
            ))
            logger.info(
                "[to_expanded] %r → app_names=[%r] type=%s subject=%r",
                req.app, req.app, req.type, req.subject,
            )

        return ExpandedRequirements(requirements=expanded)

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _call_llm_v2(self, prompt: str) -> str:
        """V2: Structured CoT variant — larger token budget, no guided_json.

        Think lines precede the JSON array, so guided_json (which constrains the
        entire output to be a JSON array) must be omitted.  _parse() handles the
        mixed Think+JSON output via regex extraction of the last [...] block.
        max_tokens is raised to 260 to accommodate the Think line(s) + JSON.
        """
        resp = self._session.post(
            f"{self._base_url}{self._llm_path}",
            json={
                "prompt": prompt,
                "max_tokens": 180,
                "stop": ["\n\n", "Task:", "\n6."],
                "repetition_penalty": 1.1,
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json().get("text", "")

    def _call_llm_v1_retry(self, prompt: str) -> str:
        """V1 fallback: direct JSON output with guided_json, no Think lines.

        Used when V2 parse returns empty (e.g. repetition loop).
        guided_json forces the entire output to be a valid JSON array,
        so _parse() reliably finds the [...] block.
        """
        resp = self._session.post(
            f"{self._base_url}{self._llm_path}",
            json={
                "prompt": prompt,
                "max_tokens": 80,
                "stop": ["\n\n", "Task:"],
                "guided_json": _GUIDED_JSON_SCHEMA,
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json().get("text", "")

    # ── Public API (V2) ────────────────────────────────────────────────────────

    def extract_v2(self, task: str) -> TaskRequirements:
        """V2 extract: uses Structured CoT prompt (_SYSTEM_PROMPT_V2).

        Produces the same TaskRequirements output as extract(), but prompts the
        model to reason step-by-step (Think lines) before emitting JSON.
        Results are cached separately from extract() to allow side-by-side
        comparison on the same task string.
        """
        if not task:
            return TaskRequirements()

        cache_key = f"__v2__{task}"
        if cache_key in self._cache:
            logger.debug("TaskExtractor v2 cache hit: %.60s", task)
            return self._cache[cache_key]

        task_clean = _clean_task(task)
        app_labels = _get_installed_app_labels(
            adb_path=self._adb_path,
            device_serial=self._device_serial,
        )
        installed_app_list = "\n".join(app_labels)
        prompt = (
            _SYSTEM_PROMPT_V2.replace("{installed_app_list}", installed_app_list)
            + f"\nTask: {task_clean}\n"
        )
        try:
            raw = self._call_llm_v2(prompt)
            logger.info("TaskExtractor V2 LLM raw: %s", raw)
            reqs = self._parse(raw)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("TaskExtractor.extract_v2 failed (%s); returning empty.", exc)
            reqs = TaskRequirements()

        if not reqs.requirements:
            logger.warning("TaskExtractor V2 parse empty; retrying with V1 fallback.")
            prompt_v1 = _SYSTEM_PROMPT_V1_RETRY + f"\nTask: {task_clean}\n"
            try:
                raw_v1 = self._call_llm_v1_retry(prompt_v1)
                logger.info("TaskExtractor V1 retry raw: %s", raw_v1)
                reqs = self._parse(raw_v1)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("TaskExtractor V1 retry failed (%s); keeping empty.", exc)

        self._cache[cache_key] = reqs
        logger.info(
            "TaskExtractor V2: task='%.60s' → %d req(s): %s",
            task,
            len(reqs.requirements),
            [(r.app, r.subject, r.type) for r in reqs.requirements],
        )
        return reqs

    def _parse(self, text: str) -> TaskRequirements:
        """Parse LLM output into TaskRequirements."""
        matches = list(re.finditer(r'\[.*\]', text, re.DOTALL))
        data = None
        for m in reversed(matches):
            try:
                candidate = json.loads(m.group())
                if isinstance(candidate, list):
                    data = candidate
                    break
            except (json.JSONDecodeError, ValueError):
                continue
        if data is None:
            logger.warning("TaskExtractor: no valid JSON array in: %.200s", text)
            return TaskRequirements()

        requirements = []
        for item in data:
            if not isinstance(item, dict):
                continue
            app     = str(item.get("app",     "")).strip()
            subject = str(item.get("subject", "")).strip()
            typ     = str(item.get("type",    "")).strip()
            if not app or not subject or not typ:
                continue
            if typ not in _VALID_TYPE_NAMES:
                logger.warning("TaskExtractor: unknown type '%s'; skipping.", typ)
                continue
            requirements.append(AppRequirement(app=app, subject=subject, type=typ))

        return TaskRequirements(requirements=requirements)


# ── Module-level helper ────────────────────────────────────────────────────────

def _clean_task(task: str) -> str:
    """Strip Chinese hint annotations and normalize newlines."""
    cleaned = re.sub(r'[。.，,]?\s*提示[：:][^\n。.]*', '', task).strip()
    return re.sub(r'\s*\n\s*', ' ', cleaned)
