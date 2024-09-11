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

import requests

# Define the GraphQL endpoint
RAILWAY_API_URL = "https://backboard.railway.app/graphql/v2"

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

assistant = openai.beta.assistants.retrieve('asst_nnCLrbQ8YoUHZB2oh6X0nSIE')
st.write(assistant.id)

from openai.types.beta.threads import Message
from openai.types.beta.threads.runs import RunStep

@dataclass
class Msg:
    role: Literal['user', 'assistant']
    stream: Iterable[AssistantStreamEvent]

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

def scan(iter, data_fn=lambda event: event, until=AssertionError, tail=None):
    for item in iter:
        try:
            yield data_fn(item)
        except until:
            if tail: tail.append(item)
            return

def run_agent_tick(stream: AssistantEventHandler):
    for event in stream:
        match event.event:
            case 'thread.message.delta':
                final = []
                st.chat_message('assistant').write_stream(
                   scan(chain([event], stream), message_content, tail=final))
                if final:
                    st.write(final)
                    st.session_state.message.append(final[0])
            case 'thread.run.step.delta':
                step = None
                for _ in scan(stream, step_delta):
                    step = stream.current_run_step_snapshot
                if step:
                    st.session_state.messages.append(step)
                    show_history_item(step)
            case 'thread.run.step.completed':
                step = event.data
                if step.type == 'message_creation':
                    st.session_state.messages.append(stream.current_message_snapshot)
            case 'thread.run.failed':
                st.chat_message('⛔️').write(event.data.last_error)  
    final_run = stream.get_final_run()
    return final_run

def run_agent_loop(*, run: Run = None, stream = None):
    with st.spinner('Processing...'):
        if stream:
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
        run_agent_loop(stream=stream)
elif thread:
    last_runs = openai.beta.threads.runs.list(thread_id=thread.id, order='desc', limit=1).data
    if last_runs:
        run_agent_loop(run=last_runs[0])
