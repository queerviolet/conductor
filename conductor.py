from dataclasses import dataclass, field, fields
from typing import AsyncIterable, Generic, Iterable, Iterator, List, Literal, Optional, TypeVar, override
from openai import AssistantEventHandler, OpenAI, AsyncOpenAI, Stream
from openai.types.beta import AssistantStreamEvent
from openai.types.beta.threads import Text, TextDelta
from openai.types.beta.threads.runs import RunStep, RunStepDelta

from pydantic import TypeAdapter
import streamlit as st

from langchain_experimental.openai_assistant import OpenAIAssistantRunnable
from langchain.agents import AgentExecutor
from langchain.tools import tool



openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@tool
def query(query: str):
    """
    query the railway graphql endpoint
    """
    return "ok"

tools = [query]

@st.cache_data
def railway_api() -> str:
    with open('railway.graphql') as idl: return idl.read()

# assistant = OpenAIAssistantRunnable(assistant_id=assistant_id(), client=openai, tools=tools, as_agent=True)
# agent = AgentExecutor(agent=assistant, tools=tools)


st.title("Conductor")
QUERY = {
    "type": "function",
    "function": {
        "name": "query",
        "description": "Query the Railway GraphQL API",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "GraphQL query to execute",
                },
                "variables": {
                    "type": "object",
                    "description": "Variables to pass to the query",
                },
            },
            "required": ["query", "variables"]
        }
    }
}

# assistant = openai.beta.assistants.create(
#     name="conductor",
#     instructions=f"""You are Conductor, an assistant for the Railway deployment platform.
#     You will be provided with the GraphQL schema for the Railway API. You are a GraphQL expert. When the user asks a question or asks for a task to be performed, provide a graphql query or mutation to handle the user's request.

#     Railway GraphQL Schema:

#     ```graphql
#     {railway_api()}
#     ```
#     """,
#     tools=[QUERY],
#     model="gpt-4o",
# )
from langchain_core.agents import AgentFinish

# def execute_agent(agent: OpenAIAssistantRunnable, tools, input: dict):
#     tool_map = {tool.name: tool for tool in tools}
#     response = agent.stream(input)
#     st.write(response)
#     while not isinstance(response, AgentFinish):
#         tool_outputs = []
#         for action in response:
#             st.write(action)
#             tool_output = tool_map[action.tool].invoke(action.tool_input)
#             tool_outputs.append({"output": tool_output, "tool_call_id": action.tool_call_id})
#         response = agent.invoke(
#             {
#                 "tool_outputs": tool_outputs,
#                 "run_id": action.run_id,
#                 "thread_id": action.thread_id
#             }
#         )

#     return response

T = TypeVar('T')

class Cursor(Generic[T]):
    iter: Iterator[T]
    _prepend: List[T]

    def __init__(self, iterable: Iterator[T] | Iterable[T]):
        self.iter = iter(iterable)
        self._prepend = []

    def scan(self, data_fn=lambda event: event, until=AssertionError):
        for evt in self:
            try:
                yield data_fn(evt)
            except until:
                self.unpop(evt)
                return


    def __iter__(self):
        yield from self._drain()
        for item in self.iter:
            yield item
            yield from self._drain()

    def _drain(self):
        while self._prepend:
            yield self._prepend.pop(0)

    def unpop(self, item: T):
        self._prepend.append(item)
        return self

def message_content(event: AssistantStreamEvent):
    assert event.event == 'thread.message.delta'
    part = event.data.delta.content[0].text.value
    return part

def function_name(event: AssistantStreamEvent):
    assert event.event == 'thread.run.step.delta'
    data = event.data.delta.step_details.tool_calls[0].function.name
    assert data
    return data

def function_arguments(event: AssistantStreamEvent):
    assert event.event == 'thread.run.step.delta'
    data = event.data.delta.step_details.tool_calls[0].function.arguments
    assert data
    return data

def show_thoughts(stream: Iterable[AssistantStreamEvent]):
    cursor = Cursor(stream)
    for event in cursor:
        match event.event:
            case 'thread.message.delta':
                cursor.unpop(event)
                st.write_stream(cursor.scan(message_content))
            case 'thread.run.step.delta':
                cursor.unpop(event)
                st.write_stream(cursor.scan(function_name))
                st.write_stream(cursor.scan(function_arguments))
            case 'thread.run.failed':
                st.write(event.data)
            case _:
                st.write(event.event)

assistant = openai.beta.assistants.retrieve('asst_nnCLrbQ8YoUHZB2oh6X0nSIE')
st.write(assistant.id)

from streamlit_javascript import st_javascript
import streamlit.components.v1 as components

from openai.types.beta.assistant_stream_event import ThreadRunStepCompleted, ThreadMessageDelta, ThreadMessageCreated
from openai.types.beta.threads import MessageDeltaEvent, MessageDelta, Message, MessageContentDelta
from openai.types.beta.threads.text_delta_block import TextDeltaBlock
from openai.types.beta.threads.runs import RunStepDelta, RunStepDeltaEvent, RunStep

def get_hash():
    x = st_javascript(f"""window.parent.location.hash""")
    if isinstance(x, str):
        return x[1:]
    return None

def set_hash(hash):
    hash = '#' + hash
    components.html(f"""<script>
    const script = parent.document.createElement('script');
    script.textContent = `window.location.hash = {hash!r}; document.currentScript.remove();`;
    parent.document.body.appendChild(script);
    </script>""")

@dataclass
class Msg:
    role: Literal['user', 'assistant']
    stream: Iterable[AssistantStreamEvent]

event_adapter = TypeAdapter(list[AssistantStreamEvent])
def to_msg(message: Message):
    event_data = [
        {
            'event': 'thread.message.delta',
            'data': {
                'id': message.id,
                'object': 'thread.message.delta',
                'delta': {
                    'content': [{
                        'index': i,                                
                        **content.model_dump(),
                    }]
                },
            }
        } for i, content in enumerate(message.content)
    ]       
    stream = event_adapter.validate_python(event_data)
    return Msg(role=message.role, stream=stream)

thread_id = get_hash()
st.write('thread_id=', thread_id)
if not thread_id:
    thread = openai.beta.threads.create()
    st.write('id=', thread.id)
    set_hash(thread.id)
    st.session_state.messages = []
    st.session_state.thread_id = thread.id
else:
    thread = openai.beta.threads.retrieve(thread_id)
    if 'messages' not in st.session_state or st.session_state.thread_id != thread.id:
        st.session_state.messages = []
        st.session_state.thread_id = thread.id
        # Load messages from OpenAI
        for message in openai.beta.threads.messages.list(thread.id, order='asc'):            
            st.session_state.messages.append(to_msg(message))

def render_message(message: Msg):
    with st.chat_message(message.role):
        show_thoughts(message.stream)

# Show message history
for message in st.session_state.messages:
    render_message(message)

if prompt := st.chat_input("🛤️"):
    message = openai.beta.threads.messages.create(thread.id, content=prompt, role='user')
    user_msg = to_msg(message)
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant"):
        # run = openai.beta.threads.create_and_run_stream(assistant_id=assistant.id, tools=[QUERY], thread_id=thread.id)
        run = openai.beta.threads.runs.stream(thread_id=thread.id, assistant_id=assistant.id, tools=[QUERY])
        while run:
            with run as stream:
                run = None
                show_thoughts(stream)
                steps = stream.get_final_run_steps()
                last_run = stream.get_final_run()
                st.write(steps)                
                for step in steps:
                    if hasattr(step.step_details, 'tool_calls'):
                        calls = step.step_details.tool_calls
                        # openai.beta.threads.runs.submit_tool_outputs()
                        run = openai.beta.threads.runs.submit_tool_outputs_stream(
                            run_id = last_run.id,
                            thread_id = last_run.thread_id,
                            tool_outputs=(
                                { 'tool_call_id': call.id, 'output': 'ok' }
                                for call in calls
                            ))
                    if run: break
                last_run

