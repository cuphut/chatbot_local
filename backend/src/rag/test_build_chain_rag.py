from src.rag.chain_rag import build_rag_chain
from src.base.llm_model_openrouter import get_openrouter_llm
import time

def test_model(model_name: str, data_dir: str, question: str):
    try:
        llm = get_openrouter_llm(model_name)
        chain_rag = build_rag_chain(llm=llm, data_dir=data_dir, data_type="pdf")
        
        start_time = time.time()
        answers = chain_rag.invoke(question)
        end_time = time.time()
        
        response_time = end_time - start_time 
        
        return {
            "model_name": model_name,
            "response_time_sec": round(response_time, 2),
            "answer_content": answers
        }
    except Exception as e:
        print(f"Lỗi khi test model {model_name}: {str(e)}")
        return None

def test_main():
    data_dir = "./data_source/generative_ai/pdfs"
    llm = get_openrouter_llm("google/gemma-3-27b-it:free")
    chain_rag= build_rag_chain(llm=llm, data_dir=data_dir,data_type="pdf")
    question = "Top 10 công ty outsouring?"
    answers = chain_rag.invoke(question)
    print("\n\n".join([doc.page_content for doc in answers]))

if __name__ == "__main__":
    data_dir = "./data_source/generative_ai/pdfs"
    question = "Tony Toàn là ai?"
    
    model_list = [
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-4-maverick:free",
        "google/gemma-3-27b-it:free",
        "qwen/qwen2.5-vl-32b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "tngtech/deepseek-r1t-chimera:free",
        "thudm/glm-z1-9b:free",
        "thudm/glm-4-9b:free",
        "microsoft/mai-ds-r1:free",
        "nvidia/llama-3.3-nemotron-super-49b-v1:free",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
        "meta-llama/llama-4-scout:free",
        "deepseek/deepseek-v3-base:free",
        "deepseek/deepseek-chat-v3-0324:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "google/gemma-3-4b-it:free",
        "google/gemma-3-27b-it:free",
        "deepseek/deepseek-r1-zero:free",
        # Thêm model bạn muốn test
    ]
    
    results = []
    for model_name in model_list:
        print(f"Testing model: {model_name}")
        result = test_model(model_name, data_dir, question)
        if result:
            results.append(result)
    
    # In bảng tổng hợp kết quả
    print("\n\nBẢNG SO SÁNH HIỆU SUẤT")
    print("{:<40} {:<15} {:<10}".format("Model", "Time (s)", "Answer (trích dẫn)"))
    for res in results:
        print("{:<40} {:<15} {:.50}".format(res["model_name"], res["response_time_sec"], res["answer_content"][:50]))
    