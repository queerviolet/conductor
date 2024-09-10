import os
import streamlit.components.v1 as components

parent_dir = os.path.dirname(os.path.abspath(__file__))
statics_dir = os.path.join(parent_dir, "statics")
_exec_graphql = components.declare_component("exec_graphql", path=statics_dir)

def exec_graphql(json):
    _exec_graphql(json=json)