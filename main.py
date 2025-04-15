import os
import json
import datetime
import streamlit as st
from groq import Groq
from langchain.memory import ConversationBufferMemory
import whisper

class ChatMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def __str__(self):
        return f"{self.role}: {self.content}"

class Conversation:
    def __init__(self, filename):
        self.filename = filename
        self.messages = []
        self.load()

    def add_message(self, message):
        self.messages.append(message)
        self.save()

    def get_history(self):
        return self.messages

    def save(self):
        with open(self.filename, 'w') as f:
            json.dump([{"role": m.role, "content": m.content} for m in self.messages], f)

    def load(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                self.messages = [ChatMessage(**m) for m in json.load(f)]
        else:
            self.save()

class ChatPipeline:
    def __init__(self, api_key, whisper_api_key, model_name='llama3-8b-8192', history_file='chat_history.json'):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.conversation = Conversation(history_file)
        self.memory = ConversationBufferMemory()
        self.whisper_api_key = whisper_api_key

    def preprocess(self, prompt):
        # Add user message to conversation history
        self.conversation.add_message(ChatMessage('User', prompt))
        # Create full prompt with conversation history
        full_prompt = "\n".join([str(m) for m in self.conversation.get_history()])
        full_prompt += f"\nUser: {prompt}\nAssistant:"
        return full_prompt

    def _forward(self, full_prompt):
        # Send prompt to Groq API
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": full_prompt,
                }
            ],
            model=self.model_name,
        )
        return chat_completion

    def postprocess(self, chat_completion):
        # Extract the response from the chat completion
        ai_response = chat_completion.choices[0].message.content
        # Add assistant message to conversation history
        self.conversation.add_message(ChatMessage("Assistant", ai_response))
        return ai_response

    def transcribe_audio(self, audio_file):
        model = whisper.load_model("base")
        result = model.transcribe(audio_file)
        return result['text']

    def __call__(self, prompt):
        # Preprocess the input to create a full prompt with history
        full_prompt = self.preprocess(prompt)
        # Get response from the LLM
        chat_completion = self._forward(full_prompt)
        # Postprocess the response and update history
        ai_response = self.postprocess(chat_completion)

        # Check if user input is asking to transcribe audio
        if prompt.startswith("transcribe "):
            audio_file = prompt.split("transcribe ")[-1]
            transcripts = self.transcribe_audio(audio_file)
            return "\n".join(transcripts)

        return ai_response

# Streamlit app
def main():
    st.title("Chat with AI")

    # Initialize the chat pipeline only once
    if 'chat_pipeline' not in st.session_state:
        api_key = "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"  # Your Groq API key
        whisper_api_key = "Jsi3qHKgE6l4yQxcbG6WtjsgsuVKKIE5"  # Your Whisper API key

        model_name = 'llama3-8b-8192'
        history_file = f'chat_history_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        st.session_state.chat_pipeline = ChatPipeline(api_key, whisper_api_key,
                                                     model_name=model_name, history_file=history_file)

    chat_pipeline = st.session_state.chat_pipeline

    # Display chat history
    st.subheader("Chat History:")
    for message in chat_pipeline.conversation.get_history():
        st.text(str(message))

    # User input
    user_input = st.text_input("You:")

    if user_input.lower() == "stop":
        st.write("Chat ended. Thank you!")
        st.stop()

    if st.button("Send"):
        if user_input:
            # Get AI response
            ai_response = chat_pipeline(user_input)

            # Clear the input box by rerunning the app
            st.experimental_rerun()

if __name__ == "__main__":
    main()