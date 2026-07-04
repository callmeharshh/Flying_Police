import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from agent.callbacks import SafeStdOutCallbackHandler
from agent.tools import log_event, trigger_alert, query_history, query_track_positions, init_tools
from storage.event_store import EventStore
from storage.frame_index import FrameIndex
from config import OPENAI_MODEL, AGENT_MEMORY_K, AGENT_VERBOSE

load_dotenv()

PROMPT_TEMPLATE = """You are Flying Police, an AI security analyst monitoring a fixed property.
You receive frame-by-frame analysis from a surveillance drone.

For each frame you get: timestamp, location, detected objects, activity, vision description, bbox position, motion tracking context, and pre-fired alerts.

Motion tracking uses bbox center positions across frames — NOT BLIP captions. The same physical object often gets different captions (e.g. "car" then "man in car" then "police car") as it moves through the scene.

Your job:
1. Use query_track_positions to fetch recent object centers at this location
2. Compare the current bbox center to prior positions — if movement is consistent (same direction, plausible distance), it is the SAME moving object even if the label changed
3. Use query_history for semantic context if needed
4. Use log_event to record the observation — note when you identified a continuing track vs a new object
5. Use trigger_alert only for genuinely new suspicious activity — NOT when motion tracking shows is_continuing=true or positions match the same moving path

When is_continuing=true or label_changed=true on the same track_id, describe it as one object moving through the scene, not multiple separate cars.

Be concise. Call log_event when there is a meaningful observation — empty or unchanged scenes do not need logging.

You have access to these tools:
{tools}

Use this format strictly:
Question: the input
Thought: your reasoning
Action: tool name (must be one of [{tool_names}])
Action Input: input to the tool
Observation: tool result
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information
Final Answer: one sentence summary of what you observed and did

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""


class SecurityAgent:
    def __init__(self, store: EventStore, index: FrameIndex):
        init_tools(store, index)

        self._llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        self._tools = [log_event, trigger_alert, query_history, query_track_positions]

        self._memory = ConversationBufferWindowMemory(
            k=AGENT_MEMORY_K,
            memory_key="chat_history",
            input_key="input",
        )

        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

        agent = create_react_agent(
            llm=self._llm,
            tools=self._tools,
            prompt=prompt,
        )

        agent_callbacks = [SafeStdOutCallbackHandler()] if AGENT_VERBOSE else []

        self._executor = AgentExecutor(
            agent=agent,
            tools=self._tools,
            memory=self._memory,
            verbose=False,
            callbacks=agent_callbacks,
            handle_parsing_errors=True,
            max_iterations=6,
        )

    def process(
        self,
        frame_id: int,
        timestamp: str,
        location: str,
        objects: list,
        activity: str,
        description: str,
        pre_alerts: list,
        vehicle_context: dict | None = None,
        bbox: tuple | None = None,
    ) -> str:

        alert_context = (
            "Pre-fired alerts: " + "; ".join(f"[{a.rule_id}] {a.message}" for a in pre_alerts)
            if pre_alerts else "No pre-fired alerts."
        )

        if bbox is not None:
            cx = round(bbox[0] + bbox[2] / 2, 1)
            cy = round(bbox[1] + bbox[3] / 2, 1)
            bbox_context = f"Bbox: ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}) center=({cx}, {cy})"
        else:
            bbox_context = "Bbox: not available"

        if vehicle_context:
            tracking_context = (
                f"Motion tracking: track_id={vehicle_context.get('track_id')}, "
                f"is_continuing={vehicle_context.get('is_continuing')}, "
                f"is_new_entry={vehicle_context.get('is_new_entry')}, "
                f"label_changed={vehicle_context.get('label_changed')}, "
                f"prior_label={vehicle_context.get('prior_object_type')}, "
                f"current_label={vehicle_context.get('object_type')}, "
                f"frame_gap={vehicle_context.get('frame_gap')}, "
                f"center_distance={vehicle_context.get('center_distance')}px "
                f"(max={vehicle_context.get('max_match_distance')}px), "
                f"trajectory={vehicle_context.get('trajectory')}"
            )
        else:
            tracking_context = "Motion tracking: not available"

        query = (
            f"Frame {frame_id} | {timestamp} | Location: {location}\n"
            f"Objects: {objects}\n"
            f"Activity: {activity}\n"
            f"Description: {description}\n"
            f"{bbox_context}\n"
            f"{tracking_context}\n"
            f"{alert_context}\n\n"
            f"Analyze this frame. Use query_track_positions at '{location}' to compare "
            f"past object centers with the current bbox. Determine if this is the same "
            f"moving object continuing or a genuinely new one. Log the event and only "
            f"trigger alerts for new suspicious activity."
        )

        try:
            result = self._executor.invoke({"input": query})
            return result.get("output", "")
        except Exception as e:
            return f"Agent error: {e}"
