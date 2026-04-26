from langchain_core.prompts import PromptTemplate
from src.retriever import retrieve_docs
from src.config import MODEL_PROVIDER

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline


prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful travel assistant.

Give a clear, structured answer using bullet points.
Do NOT repeat the context.

Context:
{context}

Question:
{question}

Answer:"""
)

if MODEL_PROVIDER == "huggingface":
    model_id = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    model.config.tie_word_embeddings = False  # silence warning

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

else:
    raise ValueError("Only HuggingFace mode enabled")


qa_chain = prompt_template | llm


def generate_answer(query: str) -> str:
    docs = retrieve_docs(query)

    if not docs:
        return "I don't know."

    # ✅ remove duplicates
    context = "\n".join(list(set(docs)))

    result = qa_chain.invoke({
        "context": context,
        "question": query
    })

    return str(result).strip()