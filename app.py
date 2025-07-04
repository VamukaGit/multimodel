import streamlit as st
import boto3
import json
import base64
import requests
import io
from PIL import Image
import os
#import sounddevice as sd
#import soundfile as sf
import tempfile
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv()

# Configure page
st.set_page_config(
    page_title="AWS Services Integration",
    page_icon="🚀",
    layout="wide"
)

# Initialize AWS clients
@st.cache_resource
def get_aws_clients():
    try:
        bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        lambda_client = boto3.client('lambda', region_name='us-east-1')
        lex_client = boto3.client('lexv2-runtime', region_name='us-east-1')
        bedrock_agent_client = boto3.client('bedrock-agent', region_name='us-east-1')
        bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime', region_name='us-east-1')
        return bedrock, lambda_client, lex_client, bedrock_agent_client, bedrock_agent_runtime_client
    except Exception as e:
        st.error(f"Error initializing AWS clients: {str(e)}")
        return None, None, None, None, None

# Audio recording utility
def record_audio(duration=5, samplerate=16000):
    
        return None

def generate_text(prompt, bedrock_client):
    try:
        response = bedrock_client.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.7
            })
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    except Exception as e:
        return f"Error generating text: {str(e)}"

def generate_image(prompt, bedrock_client):
    try:
        response = bedrock_client.invoke_model(
            modelId="stability.stable-diffusion-xl-v1",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 10,
                "seed": 42,
                "steps": 50
            })
        )
        result = json.loads(response['body'].read())
        image_data = base64.b64decode(result['artifacts'][0]['base64'])
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        return f"Error generating image: {str(e)}"

def call_lambda_summarize(text, lambda_client):
    try:
        payload = {"body": json.dumps({"text": text})}
        response = lambda_client.invoke(
            FunctionName='summarize_lambda',
            InvocationType='RequestResponse',
            Payload=json.dumps(payload),
        )
        response_payload = json.load(response['Payload'])
        return json.loads(response_payload['body']).get("summary", "No summary returned")
    except Exception as e:
        return f"Error calling Lambda: {str(e)}"

def call_api_gateway_translate(text, direction):
    try:
        url = "https://4tud9deny0.execute-api.us-east-1.amazonaws.com/translationstage"
        payload = {"body": json.dumps({"text": text, "direction": direction})}
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            inner_body = json.loads(json.loads(response.text)["body"])
            return inner_body.get("translation", "Translation not found")
        else:
            return f"API Gateway error: {response.status_code}"
    except Exception as e:
        return f"Error calling API Gateway: {str(e)}"

def process_audio_with_lex(audio_path, lex_client):
    try:
        bot_id = 'ZTEA8D6PJD'
        bot_alias_id = 'TSTALIASID'
        locale_id = 'en_US'
        session_id = 'streamlit-session'
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        response = lex_client.recognize_utterance(
            botId=bot_id,
            botAliasId=bot_alias_id,
            localeId=locale_id,
            sessionId=session_id,
            requestContentType='audio/l16; rate=16000; channels=1',
            responseContentType='audio/mpeg',
            inputStream=audio_bytes
        )
        return response.get("inputTranscript", "No text recognized"), response.get("audioStream")
    except Exception as e:
        return f"Error processing audio with Lex: {str(e)}", None

def process_image_with_ocr(image_path):
    try:
        # For now, we'll use a simple approach
        # In a real implementation, you'd use libraries like pytesseract, easyocr, or AWS Textract
        
        # Load and display the image
        image = Image.open(image_path)
        
        # Since we don't have OCR libraries installed, we'll return a placeholder
        # This would normally extract text from the image
        return "Note: OCR functionality not implemented yet. To add OCR capabilities, install pytesseract or use AWS Textract service. For now, you can describe what you see in the image and ask questions about it."
        
    except Exception as e:
        return f"Error processing image: {str(e)}"

def get_all_bedrock_agents(bedrock_agent_client):
    try:
        # List all agents
        response = bedrock_agent_client.list_agents()
        agents_data = []
        
        for agent in response.get('agentSummaries', []):
            agent_id = agent.get('agentId')
            
            try:
                # Get detailed information about each agent
                agent_details = bedrock_agent_client.get_agent(agentId=agent_id)
                
                agents_data.append({
                    'Agent ID': agent_id,
                    'Agent Name': agent.get('agentName', 'Unknown'),
                    'Status': agent.get('agentStatus', 'Unknown'),
                    'Foundation Model': agent_details.get('agent', {}).get('foundationModel', 'Unknown'),
                    'Instructions': agent_details.get('agent', {}).get('instruction', 'No instruction available'),
                    'Description': agent.get('description', 'No description available'),
                    'Created At': str(agent.get('createdAt', 'Unknown')),
                    'Updated At': str(agent.get('updatedAt', 'Unknown'))
                })
            except Exception as e:
                # If we can't get details for a specific agent, add basic info
                agents_data.append({
                    'Agent ID': agent_id,
                    'Agent Name': agent.get('agentName', 'Unknown'),
                    'Status': agent.get('agentStatus', 'Unknown'),
                    'Foundation Model': 'Error fetching details',
                    'Instructions': f'Error fetching instructions: {str(e)}',
                    'Description': agent.get('description', 'No description available'),
                    'Created At': str(agent.get('createdAt', 'Unknown')),
                    'Updated At': str(agent.get('updatedAt', 'Unknown'))
                })
        
        return {
            'success': True,
            'agents': agents_data
        }
    
    except Exception as e:
        return {
            'success': False,
            'message': f"Error retrieving agents: {str(e)}"
        }

def get_bedrock_agent_details(agent_name, bedrock_agent_client):
    try:
        # List all agents to find the one with matching name
        response = bedrock_agent_client.list_agents()
        
        for agent in response.get('agentSummaries', []):
            if agent.get('agentName', '').lower() == agent_name.lower():
                agent_id = agent.get('agentId')
                
                # Get detailed information about the agent
                agent_details = bedrock_agent_client.get_agent(agentId=agent_id)
                
                return {
                    'success': True,
                    'agent_id': agent_id,
                    'agent_name': agent.get('agentName'),
                    'agent_status': agent.get('agentStatus'),
                    'description': agent.get('description', 'No description available'),
                    'created_at': agent.get('createdAt'),
                    'updated_at': agent.get('updatedAt'),
                    'foundation_model': agent_details.get('agent', {}).get('foundationModel', 'Unknown'),
                    'instruction': agent_details.get('agent', {}).get('instruction', 'No instruction available')
                }
        
        return {
            'success': False,
            'message': f"Agent with name '{agent_name}' not found"
        }
    
    except Exception as e:
        return {
            'success': False,
            'message': f"Error retrieving agent details: {str(e)}"
        }

def get_agent_aliases(agent_id, bedrock_agent_client):
    try:
        response = bedrock_agent_client.list_agent_aliases(agentId=agent_id)
        aliases = []
        
        for alias in response.get('agentAliasSummaries', []):
            aliases.append({
                'alias_id': alias.get('agentAliasId'),
                'alias_name': alias.get('agentAliasName'),
                'alias_status': alias.get('agentAliasStatus'),
                'description': alias.get('description', 'No description')
            })
        
        return {
            'success': True,
            'aliases': aliases
        }
    
    except Exception as e:
        return {
            'success': False,
            'message': f"Error retrieving agent aliases: {str(e)}"
        }

def invoke_bedrock_agent(agent_id, agent_alias_id, session_id, input_text, bedrock_agent_runtime_client):
    try:
        # Validate inputs
        if not agent_id or not agent_alias_id:
            return {
                'success': False,
                'message': f"Invalid agent configuration - Agent ID: {agent_id}, Alias ID: {agent_alias_id}"
            }
        
        st.info(f"Invoking agent {agent_id} with alias {agent_alias_id}")
        
        response = bedrock_agent_runtime_client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=input_text
        )
        
        # Process the response stream
        event_stream = response['completion']
        response_text = ""
        
        for event in event_stream:
            if 'chunk' in event:
                chunk = event['chunk']
                if 'bytes' in chunk:
                    response_text += chunk['bytes'].decode('utf-8')
        
        return {
            'success': True,
            'response': response_text,
            'session_id': session_id
        }
    
    except Exception as e:
        error_msg = str(e)
        if "ResourceNotFoundException" in error_msg:
            return {
                'success': False,
                'message': f"Agent or alias not found. Please check:\n- Agent ID: {agent_id}\n- Agent Alias ID: {agent_alias_id}\n- Make sure the agent is deployed and the alias exists.\nOriginal error: {error_msg}"
            }
        else:
            return {
                'success': False,
                'message': f"Error invoking agent: {error_msg}"
            }

def process_pdf_with_ocr(pdf_path):
    try:
        # Extract text from PDF using PyPDF2
        reader = PdfReader(pdf_path)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        if not text.strip():
            return "Error: No text found in PDF. This might be a scanned document requiring OCR."
        
        return text.strip()
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def main():
    st.title("\U0001F680 AWS Services Integration App")

    bedrock_client, lambda_client, lex_client, bedrock_agent_client, bedrock_agent_runtime_client = get_aws_clients()
    if not all([bedrock_client, lambda_client, lex_client, bedrock_agent_client, bedrock_agent_runtime_client]):
        st.error("AWS client init failed.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["\U0001F4DD Text Input", "\U0001F3A4 Audio Input", "\U0001F680 PDF Reader", "\U0001F4DD0 Image Reader", "\U0001F4C8 AgentDetail"])

    with tab1:
        st.header("Text Input Processing")
        user_input = st.text_area("Enter your text:", height=150)
        direction = st.selectbox("Translation direction:", ["auto-en", "en-hi", "hi-en", "en-es", "es-en"])

        if st.button("Process Text", type="primary"):
            if user_input:
                user_input_lower = user_input.lower()
                if 'generate image' in user_input_lower:
                    prompt = user_input.replace('generate image', '').strip()
                    result = generate_image(prompt or "A beautiful landscape", bedrock_client)
                    st.image(result) if isinstance(result, Image.Image) else st.error(result)
                elif 'summarize' in user_input_lower:
                    text = user_input.replace('summarize', '').strip(':').strip()
                    result = call_lambda_summarize(text or "Please provide text", lambda_client)
                    st.success(result)
                elif 'translate' in user_input_lower:
                    text = user_input.replace('translate', '').strip(':').strip()
                    result = call_api_gateway_translate(text or "Hello world", direction)
                    st.success(result)
                else:
                    result = generate_text(user_input, bedrock_client)
                    st.write(result)
            else:
                st.warning("Enter some text first.")

    with tab2:
        st.header("Audio Input Processing")
        
        # Add file upload option as alternative
        st.subheader("Option 1: Upload Audio File")
        uploaded_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'ogg'])
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.write(uploaded_file.read())
            temp_file.close()
            
            if st.button("Process Uploaded Audio"):
                with st.spinner("Processing audio with Lex..."):
                    recognized_text, audio_response = process_audio_with_lex(temp_file.name, lex_client)
                    if not recognized_text.startswith("Error"):
                        st.success(f"Recognized: {recognized_text}")
                        if audio_response:
                            audio_data = audio_response.read()
                            st.audio(audio_data, format='audio/mp3')
                    else:
                        st.error(recognized_text)
        
        st.subheader("Option 2: Record Audio")
        duration = st.slider("Recording Duration (seconds)", 1, 10, 5)
        if st.button("Record and Process Audio"):
            with st.spinner("Recording and sending to Lex..."):
                audio_path = record_audio(duration)
                if audio_path is not None:
                    recognized_text, audio_response = process_audio_with_lex(audio_path, lex_client)
                    if not recognized_text.startswith("Error"):
                        st.success(f"Recognized: {recognized_text}")
                        if audio_response:
                            audio_data = audio_response.read()
                            st.audio(audio_data, format='audio/mp3')
                    else:
                        st.error(recognized_text)
                else:
                    st.warning("Recording failed. Please try uploading an audio file instead.")

    with tab3:
        st.header("PDF Question Answering")
        st.subheader("Upload PDF File")
        uploaded_pdf = st.file_uploader("Upload PDF file", type=['pdf'])

        if uploaded_pdf is not None:
            # Save uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_file.write(uploaded_pdf.read())
            temp_file.close()

            # Extract text from PDF
            if st.button("Extract Text from PDF"):
                with st.spinner("Extracting text from PDF..."):
                    pdf_text = process_pdf_with_ocr(temp_file.name)
                    if not pdf_text.startswith("Error"):
                        st.success("PDF text extracted successfully!")
                        st.session_state['pdf_text'] = pdf_text
                        
                        # Show first 500 characters as preview
                        st.subheader("PDF Content Preview:")
                        st.text_area("Content preview", pdf_text[:500] + "..." if len(pdf_text) > 500 else pdf_text, height=200, disabled=True)
                    else:
                        st.error(pdf_text)
            
            # Ask questions about the PDF
            if 'pdf_text' in st.session_state:
                st.subheader("Ask Questions about the PDF")
                question = st.text_input("Enter your question about the PDF:")
                
                if st.button("Get Answer") and question:
                    with st.spinner("Generating answer..."):
                        prompt = f"Based on the following PDF content, please answer the question.\n\nPDF Content:\n{st.session_state['pdf_text']}\n\nQuestion: {question}\n\nAnswer:"
                        answer = generate_text(prompt, bedrock_client)
                        st.success("Answer:")
                        st.write(answer)

    with tab4:
        st.header("Image Analysis")
        st.subheader("Upload Image File")
        uploaded_image = st.file_uploader("Upload Image file", type=['jpg', 'jpeg', 'png'])

        if uploaded_image is not None:
            # Display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Save uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file.write(uploaded_image.getvalue())
            temp_file.close()

            # Convert image to base64 for Claude Vision
            def encode_image_to_base64(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')

            # Ask questions about the image using Claude Vision
            st.subheader("Ask Questions about the Image")
            question = st.text_input("Enter your question about the image:")

            if st.button("Analyze Image") and question:
                with st.spinner("Analyzing image..."):
                    try:
                        # Encode image to base64
                        base64_image = encode_image_to_base64(temp_file.name)
                        
                        # Use Claude 3.5 Sonnet with vision capabilities
                        response = bedrock_client.invoke_model(
                            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
                            contentType="application/json",
                            accept="application/json",
                            body=json.dumps({
                                "anthropic_version": "bedrock-2023-05-31",
                                "messages": [{
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": question},
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "image/jpeg",
                                                "data": base64_image
                                            }
                                        }
                                    ]
                                }],
                                "max_tokens": 1000,
                                "temperature": 0.7
                            })
                        )
                        result = json.loads(response["body"].read())
                        answer = result["content"][0]["text"]
                        st.success("Analysis Result:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error analyzing image: {str(e)}")
                        st.info("Note: Image analysis requires Claude 3.5 Sonnet with vision capabilities.")
            
            # Alternative: Extract text from image (OCR)
            st.subheader("Extract Text from Image (OCR)")
            if st.button("Extract Text from Image"):
                with st.spinner("Extracting text from image..."):
                    image_text = process_image_with_ocr(temp_file.name)
                    if not image_text.startswith("Error"):
                        st.success("Image processing completed!")
                        st.info(image_text)
                    else:
                        st.error(image_text)
    with tab5:
        st.header("🤖 Bedrock Agent Management")
        
        # Section 1: Get All Agents
        st.subheader("� All Available Agents")
        
        if st.button("🔄 Load All Agents"):
            with st.spinner("Loading all agents..."):
                all_agents = get_all_bedrock_agents(bedrock_agent_client)
                
                if all_agents['success']:
                    st.session_state['all_agents'] = all_agents['agents']
                    st.success(f"Found {len(all_agents['agents'])} agents!")
                else:
                    st.error(all_agents['message'])
        
        # Display agents table if available
        if 'all_agents' in st.session_state:
            agents_df = st.session_state['all_agents']
            
            if agents_df:
                st.subheader(f"📋 Agents Summary ({len(agents_df)} total)")
                
                # Create a simplified table for better display
                display_df = []
                for agent in agents_df:
                    display_df.append({
                        'Agent Name': agent['Agent Name'],
                        'Instructions': agent['Instructions']
                    })
                
                # Display the table
                st.dataframe(display_df, use_container_width=True)
                
                # Section 2: Detailed Agent Information
                st.subheader("🔍 Detailed Agent Information")
                
                # Agent selection dropdown
                agent_names = [f"{agent['Agent Name']} ({agent['Agent ID']})" for agent in agents_df]
                selected_agent_display = st.selectbox("Select an agent to view details:", ["None"] + agent_names)
                
                if selected_agent_display != "None":
                    # Extract agent ID from the selection
                    selected_agent_id = selected_agent_display.split("(")[1].split(")")[0]
                    selected_agent = next((agent for agent in agents_df if agent['Agent ID'] == selected_agent_id), None)
                    
                    if selected_agent:
                        # Display detailed information
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Agent ID", selected_agent['Agent ID'])
                            st.metric("Agent Name", selected_agent['Agent Name'])
                            st.metric("Status", selected_agent['Status'])
                            st.metric("Foundation Model", selected_agent['Foundation Model'])
                        
                        with col2:
                            st.write("**Description:**")
                            st.write(selected_agent['Description'])
                            
                            st.write("**Created At:**")
                            st.write(selected_agent['Created At'])
                            
                            st.write("**Updated At:**")
                            st.write(selected_agent['Updated At'])
                        
                        # Display full instructions
                        st.subheader("📋 Complete Agent Instructions")
                        st.text_area("Instructions", selected_agent['Instructions'], height=200, disabled=True)
                        
                        # Get and display agent aliases
                        st.subheader("🔗 Available Agent Aliases")
                        if st.button("🔍 Load Agent Aliases"):
                            aliases_result = get_agent_aliases(selected_agent['Agent ID'], bedrock_agent_client)
                            if aliases_result['success']:
                                if aliases_result['aliases']:
                                    st.session_state['agent_aliases'] = aliases_result['aliases']
                                    st.success(f"Found {len(aliases_result['aliases'])} aliases")
                                else:
                                    st.warning("No aliases found for this agent")
                            else:
                                st.error(aliases_result['message'])
                        
                        # Display aliases if available
                        if 'agent_aliases' in st.session_state:
                            for alias in st.session_state['agent_aliases']:
                                with st.expander(f"Alias: {alias['alias_name']} ({alias['alias_id']})"):
                                    st.write(f"**Status:** {alias['alias_status']}")
                                    st.write(f"**Description:** {alias['description']}")
                                    if st.button(f"Use Alias {alias['alias_id']}", key=f"alias_{alias['alias_id']}"):
                                        st.session_state['selected_agent'] = {
                                            'agent_id': selected_agent['Agent ID'],
                                            'agent_name': selected_agent['Agent Name'],
                                            'agent_alias_id': alias['alias_id'],
                                            'instructions': selected_agent['Instructions']
                                        }
                                        st.success(f"Selected alias {alias['alias_id']} for interaction!")
                        
                        # Store selected agent for invocation
                        if st.button("✅ Select This Agent for Interaction"):
                            st.session_state['selected_agent'] = {
                                'agent_id': selected_agent['Agent ID'],
                                'agent_name': selected_agent['Agent Name'],
                                'agent_alias_id': '5OSGEVJVLE',  # Default alias ID
                                'instructions': selected_agent['Instructions']
                            }
                            st.success(f"Agent '{selected_agent['Agent Name']}' selected for interaction!")
                            st.warning("Using default alias ID '5OSGEVJVLE'. If this doesn't work, load and select a specific alias above.")
            else:
                st.info("No agents found in your AWS account.")
        
        # Section 3: Predefined Agent (fallback)
        st.subheader("🎯 Quick Access - Predefined Agent")
        
        with st.expander("Known Agent: simple-text-agent"):
            st.write("**Agent ID:** JCIGJX9AQG")
            st.write("**Agent Alias ID:** 5OSGEVJVLE")
            st.write("**Status:** Available for interaction")
            
            if st.button("🚀 Select Predefined Agent"):
                st.session_state['selected_agent'] = {
                    'agent_id': 'JCIGJX9AQG',
                    'agent_name': 'Predefined Agent',
                    'agent_alias_id': '5OSGEVJVLE',
                    'instructions': 'Predefined agent with standard capabilities'
                }
                st.success("Predefined agent selected for interaction!")
        
        # Section 4: Agent Interaction
        if 'selected_agent' in st.session_state:
            st.subheader("💬 Interact with Selected Agent")
            
            agent_info = st.session_state['selected_agent']
            
            # Display selected agent info
            st.info(f"🤖 Selected Agent: **{agent_info.get('agent_name', 'Unknown')}** (ID: {agent_info.get('agent_id', 'Unknown')})")
            st.info(f"🔗 Using Alias ID: **{agent_info.get('agent_alias_id', 'None')}**")
            
            # Quick alias check button
            if st.button("🔍 Verify Agent & Alias"):
                with st.spinner("Verifying agent configuration..."):
                    aliases_result = get_agent_aliases(agent_info.get('agent_id'), bedrock_agent_client)
                    if aliases_result['success']:
                        if aliases_result['aliases']:
                            st.success(f"✅ Agent found with {len(aliases_result['aliases'])} aliases:")
                            current_alias = agent_info.get('agent_alias_id')
                            alias_found = False
                            
                            for alias in aliases_result['aliases']:
                                status_emoji = "✅" if alias['alias_id'] == current_alias else "ℹ️"
                                st.write(f"{status_emoji} **{alias['alias_name']}** (ID: `{alias['alias_id']}`) - Status: {alias['alias_status']}")
                                if alias['alias_id'] == current_alias:
                                    alias_found = True
                            
                            if not alias_found and current_alias:
                                st.error(f"❌ Current alias ID '{current_alias}' not found in available aliases!")
                                st.info("💡 Please select a different alias from the list above.")
                        else:
                            st.warning("⚠️ Agent found but no aliases available")
                    else:
                        st.error(f"❌ Agent verification failed: {aliases_result['message']}")
            
           
            # Input for agent interaction
            col1, col2 = st.columns([3, 1])
            
            with col1:
                user_question = st.text_area("💭 Ask your question to the agent:", height=100, placeholder="Type your question here...")
            
            with col2:
                st.write("**Session Settings:**")
                session_id = st.text_input("Session ID:", value="streamlit-agent-session")
                
                # Predefined prompts
                st.write("**Quick Prompts:**")
                if st.button("❓ What can you do?"):
                    user_question = "What are your capabilities and how can you help me?"
                if st.button("📊 Analyze data"):
                    user_question = "Can you help me analyze some data?"
                if st.button("💡 Give suggestions"):
                    user_question = "Can you provide some suggestions based on your expertise?"
            
            
            # Invoke button
            if st.button("🚀 Invoke Agent", type="primary") and user_question:
                with st.spinner("🤖 Agent is processing your request..."):
                    # Use predefined alias ID or get from agent details
                    agent_alias_id = agent_info.get('agent_alias_id', '5OSGEVJVLE')
                    
                    result = invoke_bedrock_agent(
                        agent_id=agent_info['agent_id'],
                        agent_alias_id=agent_alias_id,
                        session_id=session_id,
                        input_text=user_question,
                        bedrock_agent_runtime_client=bedrock_agent_runtime_client
                    )
                    
                    if result['success']:
                        st.success("✅ Agent Response:")
                        st.markdown(f"**Response:** {result['response']}")
                        
                        # Show session info
                        st.info(f"🔗 Session ID: {result['session_id']}")
                        
                        # Store conversation history
                        if 'conversation_history' not in st.session_state:
                            st.session_state['conversation_history'] = []
                        
                        st.session_state['conversation_history'].append({
                            'question': user_question,
                            'response': result['response'],
                            'timestamp': str(st.session_state.get('timestamp', 'Unknown'))
                        })
                    else:
                        st.error(f"❌ Error: {result['message']}")
            
            # Show conversation history
            if 'conversation_history' in st.session_state and st.session_state['conversation_history']:
                st.subheader("💬 Conversation History")
                
                for i, conv in enumerate(reversed(st.session_state['conversation_history'][-5:])):  # Show last 5 conversations
                    with st.expander(f"Conversation {len(st.session_state['conversation_history'])-i}"):
                        st.write(f"**Question:** {conv['question']}")
                        st.write(f"**Response:** {conv['response']}")
                
                if st.button("🗑️ Clear History"):
                    st.session_state['conversation_history'] = []
                    st.success("Conversation history cleared!")
        
        # Section 5: Tips and Help
        st.subheader("💡 Tips & Help")
        st.info("""
        **How to use this tab:**
        1. **Load Agents**: Click "Load All Agents" to see all available agents in your AWS account
        2. **Browse Table**: View agents in table format with ID, name, status, and instruction previews
        3. **Select Agent**: Choose an agent from the dropdown to view detailed information
        4. **Interact**: Select an agent for interaction and ask questions
        5. **Quick Prompts**: Use predefined prompts or type your own questions
        6. **History**: View your conversation history with the agent
        
        **Note:** Make sure your AWS credentials have the necessary permissions for Bedrock Agent access.
        """)
            
if __name__ == "__main__":
    main()
