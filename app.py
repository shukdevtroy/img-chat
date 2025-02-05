import os
import tempfile
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import SimpleDirectoryReader

# Function to display images
def plot_image(image):
    st.image(image, use_container_width=True)

# Function to process and get response from LLM
def get_response_from_image(image, prompt, api_key):
    # Create temporary directory for image input
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploaded image to temporary directory
        image_file_path = os.path.join(tmpdir, "uploaded_image.png")
        image.save(image_file_path)

        # Prepare the SimpleDirectoryReader
        image_documents = SimpleDirectoryReader(tmpdir).load_data()

        # Initialize the LLM
        openai_mm_llm = OpenAIMultiModal(
            model="gpt-4o", api_key=api_key, max_new_tokens=1500
        )

        # Generate a response
        response = openai_mm_llm.complete(
            prompt=prompt,
            image_documents=image_documents,
        )

        # Debug: Print the entire response to inspect its structure
        print(response)  # Remove this after you understand the structure

        # Assuming the response has a 'text' attribute or similar
        # If the response has a 'text' attribute, directly access it:
        if hasattr(response, 'text'):
            return response.text  # Return the 'text' field

        # If the response has 'message' attribute (based on OpenAI-style response):
        if hasattr(response, 'message'):
            return response.message  # Return the 'message' field

        # If nothing works, return a fallback message
        return "No response text found"

# Streamlit app interface
def main():
    # Set up title
    st.title("Image Question Answering with OpenAI LLM")
    
    # Input OpenAI API Key
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    
    if api_key:
        # Image upload
        uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
        
        if uploaded_image:
            # Open and display the uploaded image
            image = Image.open(uploaded_image)
            plot_image(image)

            # Input prompt
            prompt = st.text_area("Ask a question about the image:", placeholder="e.g. What is this image about?")
            
            # Add a button to generate the response
            if st.button("Generate Response"):
                if prompt:
                    # Get response from OpenAI
                    with st.spinner("Processing..."):
                        response = get_response_from_image(image, prompt, api_key)
                    
                    # Display only the text part of the response
                    st.subheader("Response:")
                    st.write(response)
                else:
                    st.warning("Please enter a prompt before generating a response.")

            # Option to clear the uploaded image and prompt
            if st.button("Clear"):
                st.cache_data.clear()  # Clear cache for the application

if __name__ == "__main__":
    main()