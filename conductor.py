from dataclasses import dataclass, field, fields
from typing import AsyncIterable, Generic, Iterable, Iterator, List, Literal, Optional, TypeVar, override
from openai import AssistantEventHandler, OpenAI, AsyncOpenAI, Stream
from openai.types.beta import AssistantStreamEvent
from openai.types.beta.threads import Text, TextDelta, Run
from openai.types.beta.threads.runs import RunStep, RunStepDelta

from pydantic import TypeAdapter
import streamlit as st

import os

# try:
#     OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
# except FileNotFoundError:
#     OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

openai = OpenAI()
aopenai = AsyncOpenAI()

@st.cache_data
def railway_api() -> str:
    with open('railway.graphql') as idl: return idl.read()

TOOL_RESULTS = st.session_state.setdefault('tool_results', {})

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

def step_delta(event: AssistantStreamEvent):
    assert event.event == 'thread.run.step.delta'
    data = event.data.delta.step_details.tool_calls[0].function.name
    if not data:
        data = event.data.delta.step_details.tool_calls[0].function.arguments
    return data


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


import requests

# Define the GraphQL endpoint
RAILWAY_API_URL = "https://backboard.railway.app/graphql/v2"

@st.dialog("Railway API Key needed")
def request_api_key():
    if api_key := st.text_input(label='Railway API Key', value=None):
        st.session_state.RAILWAY_API_KEY = api_key
        st.rerun()

def exec_graphql(json, key, result=None):
    with st.form(key):
        api_key = st.session_state.get('RAILWAY_API_KEY', default=None)
        api_key = st.text_input('Railway API Key', value=api_key, type='password')
        st.session_state.RAILWAY_API_KEY = api_key
        try:
            if st.form_submit_button(label="Run", disabled=not not result):
                response = requests.post(RAILWAY_API_URL, data=json, headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                })
                return response.text
        finally:
            if result and result is not True:
                st.code(result, key=f'result_{key}')

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
                result = exec_graphql(json=fn_args[len(before):-len(after)], key=id)
                if result is not None:
                    TOOL_RESULTS[id] = result
            case 'thread.run.failed':
                st.write(event.data)
            case _:
                status.write(event.event)

assistant = openai.beta.assistants.retrieve('asst_nnCLrbQ8YoUHZB2oh6X0nSIE')
st.write(assistant.id)

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
import asyncio as aio

async def load_history(thread: Thread):
    messages: list[Message | Run | RunStep] = []
    recent_runs_task = aio.create_task(load_last_runs(thread.id, messages))
    async for message in aopenai.beta.threads.messages.list(thread.id, order='asc'):
        messages.append(message)        
    await recent_runs_task
    messages.sort(key=lambda x: x.created_at)
    return messages

async def load_run_steps(thread_id, run_id, messages: list[Message | Run | RunStep]):
    async for step in aopenai.beta.threads.runs.steps.list(run_id=run_id, thread_id=thread_id):
        if step.type == 'tool_calls': messages.append(step)

async def load_last_runs(thread_id, messages: list[Message | Run | RunStep]):
    load_step_tasks = []
    async for run in aopenai.beta.threads.runs.list(thread_id, order='desc', limit=10):
        load_step_tasks.append(aio.create_task(load_run_steps(thread_id, run.id, messages)))
    await aio.gather(*load_step_tasks)

thread: Optional[Thread] = st.session_state.get('thread', default=None)
thread_id = st.query_params.get('thread_id', None)
if getattr(thread, 'id', None) != thread_id:
    if thread_id:
        thread = openai.beta.threads.retrieve(thread_id)
        st.session_state.thread = thread
        st.session_state.messages = aio.run(load_history(thread))        

def render_message(message: Msg):
    with st.chat_message(message.role):
        show_thoughts(message.stream)

from html import escape
from streamlit.delta_generator import DeltaGenerator

# class RunStatusLine:
#     last: Optional['RunStatusLine'] = None
#     container: DeltaGenerator

#     def __init__(self, run: Optional[Run]):
#         self.container = st.empty()
#         self.text = ''
#         self.run = run
#         self.update(run)
#         RunStatusLine.last = self
    
#     def update(self, run: Optional[Run] = None, text: Optional[str] = None):
#         if run:
#             self.run = run
#         else:
#             run = self.run
#         if not run:
#             content = ''
#         else:        
#             ended_at = getattr(run, f'{run.status}_at', None)
#             if ended_at:
#                 duration = ended_at - run.created_at
#                 duration_str = f' in {duration}ms'
#             else: duration_str = ''
#             content = escape(f'{run.id} {run.status}{duration_str}')
#             if text:
#                 content = f"<pre style='color: blue'>{escape(self.text)}</pre>{content}"
#         self.container.html(f"""
#         <div style='font-size: 75%; text-align: right'>{content}</div>
#         """)

import json

from openai.types.beta.threads.runs.function_tool_call import FunctionToolCall

def show_history_item(message: Message | Run | RunStep):
    match message:
        case Message(role=role, content=content):
            if content:
                st.chat_message(role).write(content[0].text.value)
        case RunStep() as step:
            if step.type == 'message_creation': return
            calls = [c for c in step.step_details.tool_calls if c.type == 'function']
            if not calls: return
            with st.chat_message('⚙️'): #꩟ ⫸ ∭
                for call in calls:
                    if call.type != 'function': continue
                    show_function_call(step, call)                
        case _:
            st.write(message)

def show_function_call(step: RunStep, call: FunctionToolCall):
    try:
        md = json.loads(call.function.arguments)
    except BaseException as ex:
        st.error(ex)
    if not md: return
    st.code(md['query'], wrap_lines=True)
    del md['query']
    if md: st.write(md)
    if step.status == 'in_progress':
        result = exec_graphql(json=call.function.arguments, key=call.id)
        if result is not None:
            st.code(result)
            TOOL_RESULTS[call.id] = result
    else:
        exec_graphql(json=call.function.arguments,
                     key=call.id,
                     result=TOOL_RESULTS.get(call.id, True))

# Show message history
EMPTY: list[Message | Run | RunStep] = []
for message in st.session_state.get('messages', default=EMPTY):
    # render_message(message)
    show_history_item(message)

from openai.types.beta.threads.text_content_block import TextContentBlock

def run_agent_tick(stream: AssistantEventHandler):
    # assistant_msg = Msg(role='assistant', stream=Record(stream))
    # st.session_state.messages.append(assistant_msg)
    # render_message(assistant_msg)
    cursor = Cursor(stream)
    for event in stream:
        match event.event:
            case 'thread.message.completed':
                st.session_state.messages.append(event.data)
            case 'thread.message.delta':
                cursor.unpop(event)
                st.chat_message('assistant').write_stream(cursor.scan(message_content))
            case 'thread.run.step.delta':
                step = None
                cursor.unpop(event)
                for _ in cursor.scan(step_delta):
                    current_step = stream.current_run_step_snapshot
                    if current_step: step = current_step
                if step:
                    st.session_state.messages.append(step)
                    show_history_item(step)
            case 'thread.run.failed':
                st.chat_message('⛔️').write(event.data)
    final_run = stream.get_final_run()
    return final_run

def run_agent_loop(run: Run):
    with st.spinner('Processing...'):
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

if prompt := st.chat_input("🛤️"):
    if not thread:
        thread = openai.beta.threads.create()
        st.query_params.thread_id = thread.id
        st.session_state.thread = thread
        st.session_state.messages = []

    message = openai.beta.threads.messages.create(thread.id, content=prompt, role='user')
    st.session_state.messages.append(message)
    show_history_item(message)

    with openai.beta.threads.runs.stream(thread_id=thread.id, assistant_id=assistant.id, tools=[QUERY]) as stream:
        run_agent_loop(run_agent_tick(stream))
elif thread:
    last_runs = openai.beta.threads.runs.list(thread_id=thread.id, order='desc', limit=1).data
    if last_runs:
        run_agent_loop(last_runs[0])
