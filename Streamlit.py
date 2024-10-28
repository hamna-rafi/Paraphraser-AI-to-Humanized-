


import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set the device to CPU
device = "cpu"

# Set up the Streamlit app title
st.title('Paraphraser AI-to-Humanized')

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

# Define the paraphrase function
def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=3,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)
    
    outputs = model.generate(
        input_ids,
        repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        max_length=max_length,
        diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

# Create a text input box for the user
input_text = st.text_area("Enter text to paraphrase:", height=200)

# Create a button to trigger paraphrasing
if st.button("Paraphrase"):
    if input_text:
        # Call the paraphrase function
        results = paraphrase(input_text)
        
        # Display the results
        if results:
            st.subheader("Paraphrased Results:")
            for i, result in enumerate(results, start=1):
                st.write(f"{i}. {result}")
        else:
            st.write("No paraphrased results found.")
    else:
        st.write("Please enter some text to paraphrase.")
