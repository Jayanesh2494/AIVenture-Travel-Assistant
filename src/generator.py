from langchain_core.prompts import PromptTemplate
from src.retriever import retrieve_docs
from src.config import MODEL_PROVIDER

# Hugging Face
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline


# Prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful travel assistant.
Use the following travel guide context to answer the question.
If the answer is not found, say "I don't know" — do NOT make it up.

Context:
{context}

Question:
{question}

Answer:"""
)


# ✅ Use FLAN-T5-LARGE
if MODEL_PROVIDER == "huggingface":
    model_id = "google/flan-t5-large"

    print("🚀 Loading FLAN-T5-LARGE model... (this may take time)")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

else:
    raise ValueError("Only HuggingFace mode is enabled now")


# Chain
qa_chain = prompt_template | llm


def generate_answer(query: str) -> str:
    docs = retrieve_docs(query)

    if not docs:
        return "I don't know."

    # Handle both formats
    if isinstance(docs[0], str):
        context = "\n".join(docs)
    else:
        context = "\n".join([doc.page_content for doc in docs])

    result = qa_chain.invoke({
        "context": context,
        "question": query
    })

    return str(result).strip()