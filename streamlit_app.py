import streamlit as st
from agent import agent
import json
import base64


def create_download_link(val, filename):
    b64 = base64.b64encode(val)
    return f'href="data:application/json;base64,{b64.decode()}" download="{filename}.json"'


if __name__ == '__main__':

    st.set_page_config(layout="wide", page_title="Workflow Automation")
    with open("assets/styles.css") as css:
        st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)
    
    st.header('Workflow Automation (Prompt to JSON)')

    cols = st.columns([0.45, 0.1, 0.45])

    with cols[0]:
        prompt = st.text_area(
            label='Prompt', value=None,
            placeholder='Enter workflow generation prompt.',
            label_visibility='hidden',
        )

        try:
            if st.button('Submit Prompt'):
                if prompt is None or prompt.strip() == '':
                    st.warning('Prompt should not be empty.', icon="⚠️")
                
                else:
                    with cols[2]:
                        with st.spinner("Generating JSON based on the prompt..."):
                            result = agent(prompt)
                
                        try:
                            json_object = json.loads(result)
                            json_data = json.dumps(json_object, indent=2).encode('utf-8')
                            download_tag = create_download_link(json_data, 'workflow_response')

                            st.markdown(
                            f"""
                            <a {download_tag} target="_self">
                                <div style="
                                    display: inline-block;
                                    padding: 0.5em 1em;
                                    color: #000000;
                                    background-color: #F0F0F0;
                                    border-radius: 5px;
                                    border: #000000,
                                    text-decoration: none;">
                                    Download Response JSON
                                </div>
                            </a>
                            """,
                            unsafe_allow_html=True
                            )
                            
                            st.markdown('---')
                            st.write(json_object)
                            
                        except json.JSONDecodeError:
                            st.markdown("##### JSON response could not be fetched from the server.")
                            st.markdown("*Try submitting the prompt again after 5 minutes.*")
                            
        except Exception as e:
            with cols[2]:
                st.markdown("##### JSON response could not be fetched from the server.")
                st.markdown("*Try submitting the prompt again after 5 minutes.*")

        st.image("assets/olist_schema.png")