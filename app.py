from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cpu"  

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    # temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)
    
    outputs = model.generate(
        input_ids,  repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res
text = """As the political landscape of India began to change in the early 20th century, Jinnah's vision evolved. He became increasingly aware of the distinct political identity and aspirations of Muslims in India. The formation of the All-India Muslim League in 1906 marked a significant turning point in his political career. Jinnah initially sought to foster cooperation between Hindus and Muslims but eventually recognized that the political rights of Muslims could only be secured through separate representation and autonomy."""
# paraphrase(text)
print(paraphrase(text))