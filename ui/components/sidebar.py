"""
Sidebar Navigation Component
"""
import streamlit as st

def render_sidebar():
    """Render the main application sidebar with navigation and project status."""
    st.sidebar.title("Pipeline Steps")
    pages = {
        'upload': '1. Upload & Extract',
        'prompts': '2. Configure Prompts',
        'sample_test': '3. Sample Test',
        'annotate': '4. Full Annotation',
        'organize': '5. Organize Dataset',
        'summary': '6. Summary',
        'train': '7. Train Model'
    }

    for key, label in pages.items():
        if st.sidebar.button(label, use_container_width=True):
            st.session_state.page = key

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Current Status")
    
    project_name = "None"
    if st.session_state.get('project_manager') is not None:
        project_name = st.session_state.project_manager.project_name
        
    st.sidebar.metric("Project", project_name)
    st.sidebar.metric("Frames", len(st.session_state.get('extracted_frames', [])))
    st.sidebar.metric("Prompts", len(st.session_state.get('text_prompts', [])))
    st.sidebar.metric("Annotated", len(st.session_state.get('annotations', {})))
    st.sidebar.metric("Selected", len(st.session_state.get('selected_images', set())))
