from openai import OpenAI, Stream
from openai.types.beta import AssistantStreamEvent
import streamlit as st

st.title("Conductor")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
assistant = client.beta.assistants.retrieve('asst_EvVlj7IYENH39CUEbxJucVoF')

if "thread_id" not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state["thread_id"] = thread.id
else:
    thread = client.beta.threads.retrieve(st.session_state["thread_id"])        
st.write(thread.id)

for message in client.beta.threads.messages.list(thread.id):
    with st.chat_message(message.role):
        st.markdown(message.content[0].text.value)

def format_stream(stream: Stream[AssistantStreamEvent]):
    for event in stream:
        match event.event:
            case 'thread.message.delta':
                yield event.data.delta.content[0].text.value
            case _:
                yield '![](' + event.event + ')'

if prompt := st.chat_input("🛤️"):
    message = client.beta.threads.messages.create(thread.id, content=prompt, role='user')
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.beta.threads.runs.create(thread.id, assistant_id=assistant.id, stream=True)
        st.write_stream(format_stream(stream))

