from dataclasses import dataclass, field, fields
from typing import AsyncIterable, Generic, Iterable, Iterator, List, Literal, Optional, TypeVar, override
from openai import AssistantEventHandler, OpenAI, AsyncOpenAI, Stream
from openai.types.beta import AssistantStreamEvent
from openai.types.beta.threads import Text, TextDelta, Run
from openai.types.beta.threads.runs import RunStep, RunStepDelta

from pydantic import TypeAdapter
import streamlit as st

from record import Record


openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


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

TOOL_RESULTS = {}

import requests

# Define the GraphQL endpoint
RAILWAY_API_URL = "https://backboard.railway.app/graphql/v2"

@st.dialog("Railway API Key needed")
def request_api_key():
    if api_key := st.text_input(label='Railway API Key', value=None):
        st.session_state.RAILWAY_API_KEY = api_key
        st.rerun()

def exec_graphql(json):
    if st.button(label="Run", key=json):
        api_key = st.session_state.get('RAILWAY_API_KEY', default=None)
        if not api_key:
            request_api_key()
        else:
            response = requests.post(RAILWAY_API_URL, data=json, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            })
            return response.text

from itertools import chain


def show_thoughts(stream: Iterable[AssistantStreamEvent]):
    status = st.empty()
    cursor = Cursor(stream)
    for event in cursor:
        match event.event:
            case 'thread.message.delta':
                cursor.unpop(event)
                st.write_stream(cursor.scan(message_content))
            case 'thread.run.step.delta':
                cursor.unpop(event)
                st.write_stream(cursor.scan(function_name))
                before = '<small><pre>'
                after = '</pre></small>'
                fn_args = ''.join(st.write_stream(chain(
                    [before],
                    cursor.scan(function_arguments),
                    [after])))
                id = event.data.delta.step_details.tool_calls[0].id
                # result = st.text_area(label='Result:', placeholder=''.join(fn_args), value=None, key=id)
                result = exec_graphql(json=fn_args[len(before):-len(after)])
                if result is not None:
                    TOOL_RESULTS[id] = result
            case 'thread.run.failed':
                st.write(event.data)
            case _:
                st.write(event.event)
    status.empty()

assistant = openai.beta.assistants.retrieve('asst_nnCLrbQ8YoUHZB2oh6X0nSIE')
st.write(assistant.id)

from streamlit_javascript import st_javascript
import streamlit.components.v1 as components

from openai.types.beta.assistant_stream_event import ThreadRunStepCompleted, ThreadMessageDelta, ThreadMessageCreated
from openai.types.beta.threads import MessageDeltaEvent, MessageDelta, Message, MessageContentDelta
from openai.types.beta.threads.text_delta_block import TextDeltaBlock
from openai.types.beta.threads.runs import RunStepDelta, RunStepDeltaEvent, RunStep

@dataclass
class Msg:
    role: Literal['user', 'assistant']
    stream: Iterable[AssistantStreamEvent]

event_adapter = TypeAdapter(list[AssistantStreamEvent])
def to_msg(message: Message):
    event_data = (
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
    )   
    stream = event_adapter.validate_python(event_data)
    return Msg(role=message.role, stream=stream)

def steps_to_msg(steps: list[RunStep]):
    event_data = []
    for step in steps:
        event_data.extend(step_events(step))
    stream = event_adapter.validate_python(event_data)
    return Msg(role='assistant', stream=stream)

from openai.types.beta.threads.runs.tool_call_delta_object import ToolCallDeltaObject
from openai.types.beta.threads.runs.tool_call_delta import ToolCallDelta
from openai.types.beta.threads.runs.function_tool_call_delta import FunctionToolCallDelta

def step_events(step: RunStep):
    if step.type != 'tool_calls': return
    for i, call in enumerate(step.step_details.tool_calls):
        if call.type == 'function':
            # RunStepDelta(
            #     step_details=ToolCallDeltaObject(type='tool_calls', tool_calls=[
            #         FunctionToolCallDelta(index=i, type='function', id=call.id, function=
            #     ])
            # )
            yield {
                'event': 'thread.run.step.delta',
                'data': {
                    'id': step.id,
                    'object': 'thread.run.step.delta',
                    'delta': {
                        'step_details': {
                            'type': step.type,
                            'tool_calls': [{
                                'id': call.id,
                                'index': i,
                                'type': 'function',
                                'function': {
                                    'name': call.function.name
                                }
                            }]
                        }
                    }
                }
            }
            yield {
                'event': 'thread.run.step.delta',
                'data': {
                    'id': step.id,
                    'object': 'thread.run.step.delta',
                    'delta': {
                        'step_details': {
                            'type': step.type,
                            'tool_calls': [{
                                'id': call.id,
                                'index': i,
                                'type': 'function',
                                'function': {
                                    'arguments': call.function.arguments
                                }
                            }]
                        }
                    }
                }
            }

from openai.types.beta.thread import Thread

thread: Optional[Thread] = st.session_state.get('thread', default=None)
thread_id = st.query_params.get('thread_id', None)
if getattr(thread, 'id', None) != thread_id:
    if thread_id:
        thread = openai.beta.threads.retrieve(thread_id)
        st.session_state.messages = []
        st.session_state.thread = thread
        # Load messages from OpenAI
        for message in openai.beta.threads.messages.list(thread.id, order='asc'):            
            st.session_state.messages.append(to_msg(message))
        last_runs = openai.beta.threads.runs.list(thread_id=thread.id, order='desc', limit=1).data
        if last_runs:
            run = last_runs[0]
            steps = openai.beta.threads.runs.steps.list(run_id=run.id, thread_id=thread.id)
            st.session_state.messages.append(steps_to_msg(steps.data))

def render_message(message: Msg):
    with st.chat_message(message.role):
        show_thoughts(message.stream)

# Show message history
for message in st.session_state.get('messages', []):
    render_message(message)

def run_agent_tick(stream: AssistantEventHandler):
    assistant_msg = Msg(role='assistant', stream=Record(stream))
    st.session_state.messages.append(assistant_msg)
    render_message(assistant_msg)
    return stream.get_final_run()

def run_agent_loop(*, stream: Optional[AssistantEventHandler] = None, run: Optional[Run] = None):
    assert stream or run and not (stream and run)
    if not run:
        run = run_agent_tick(stream)
    while run:
        run = handle_required_actions(run)    

def handle_required_actions(run: Run):
    if not run.required_action: return None
    calls = run.required_action.submit_tool_outputs.tool_calls
    if all(call.id in TOOL_RESULTS for call in calls):
        stream_mgr = openai.beta.threads.runs.submit_tool_outputs_stream(
            run_id = run.id,
            thread_id = run.thread_id,
            tool_outputs=(
                { 'tool_call_id': call.id, 'output': TOOL_RESULTS[call.id] }
                for call in calls
            ))
        with stream_mgr as stream:
            return run_agent_tick(stream)
    # else:
    #     st.write('waiting for')
    #     [call.id if call.id not in TOOL_RESULTS else None for call in calls]

if prompt := st.chat_input("🛤️"):
    if not thread:
        thread = openai.beta.threads.create()
        st.query_params.thread_id = thread.id
        st.session_state.thread = thread
        st.session_state.messages = []

    message = openai.beta.threads.messages.create(thread.id, content=prompt, role='user')
    user_msg = to_msg(message)
    st.session_state.messages.append(user_msg)
    render_message(user_msg)
    
    with openai.beta.threads.runs.stream(thread_id=thread.id, assistant_id=assistant.id, tools=[QUERY]) as stream:
        run_agent_loop(stream=stream)
elif thread:
    last_runs = openai.beta.threads.runs.list(thread_id=thread.id, order='desc', limit=1).data
    if last_runs:
        last_runs[0].status
        run_agent_loop(run=last_runs[0])
