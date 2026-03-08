"""
Configure Prompts Page
Separate Prompt and Class Name inputs
"""
import streamlit as st
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.logging_config import setup_logging

log = setup_logging("prompts_page")

def render_prompts_page():
    st.header("2. Configure Text Prompts")
    
    if st.session_state.project_manager is None:
        st.warning("Please create or load a project first")
        return
    
    pm = st.session_state.project_manager
    
    st.markdown("""
    **Prompt**: What SAM3 will search for (can be detailed description)  
    **Class Name**: Label used in YAML output (short, no spaces)
    """)
    
    # Use form to allow proper reset
    with st.form("add_prompt_form", clear_on_submit=True):
        col1, col2 = st.columns([2, 1])
        
        new_prompt = col1.text_input(
            "Prompt (e.g., 'person wearing red helmet')", 
            placeholder="What to detect..."
        )
        
        class_name = col2.text_input(
            "Class Name (e.g., 'worker')", 
            placeholder="Label for YAML"
        )
        
        submitted = st.form_submit_button("Add Prompt", use_container_width=True)
        
        if submitted and new_prompt:
            # Auto-generate class name if empty
            final_class = class_name if class_name else new_prompt.split()[0].lower().replace(" ", "_")
            
            # Check for duplicates
            existing_classes = [p['class_name'] for p in st.session_state.text_prompts]
            if final_class in existing_classes:
                st.error(f"Class name '{final_class}' already exists!")
            else:
                st.session_state.text_prompts.append({
                    'prompt_text': new_prompt,
                    'class_name': final_class,
                    'class_id': len(st.session_state.text_prompts)
                })
                pm.set_prompts(st.session_state.text_prompts)
                log.info(f"Added prompt: '{new_prompt}' → Class: {final_class}")
                st.rerun()
    
    # Display prompts table
    st.subheader("Current Prompts")
    
    if not st.session_state.text_prompts:
        st.info("No prompts added yet. Add prompts above.")
    else:
        # Table header
        cols = st.columns([0.5, 2, 1, 0.5])
        cols[0].write("**ID**")
        cols[1].write("**Prompt**")
        cols[2].write("**Class Name**")
        cols[3].write("**Action**")
        
        st.markdown("---")
        
        for i, prompt_data in enumerate(st.session_state.text_prompts):
            cols = st.columns([0.5, 2, 1, 0.5])
            cols[0].write(f"{i}")
            cols[1].write(prompt_data['prompt_text'])
            cols[2].write(f"`{prompt_data['class_name']}`")
            
            if cols[3].button("X", key=f"del_{i}"):
                st.session_state.text_prompts.pop(i)
                # Re-assign class IDs
                for j, p in enumerate(st.session_state.text_prompts):
                    p['class_id'] = j
                pm.set_prompts(st.session_state.text_prompts)
                log.info(f"Removed prompt: {prompt_data['prompt_text']}")
                st.rerun()
    
    # Summary
    st.markdown("---")
    st.info(f"{len(st.session_state.text_prompts)} prompts configured")
    
    # Preview YAML classes
    if st.session_state.text_prompts:
        st.subheader("YAML Preview")
        yaml_preview = "names:\n"
        for p in st.session_state.text_prompts:
            yaml_preview += f"  {p['class_id']}: {p['class_name']}\n"
        st.code(yaml_preview, language="yaml")
    
    # Next button
    if st.session_state.text_prompts and st.session_state.extracted_frames:
        st.markdown("---")
        if st.button("Next -> Sample Test", type="primary", use_container_width=True):
            st.session_state.page = 'sample_test'
            st.rerun()
