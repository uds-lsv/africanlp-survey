import sys
import time
import openai
import pandas as pd
import json
from tqdm import tqdm  # Import tqdm for the progress bar

# Set your OpenAI API key
openai.api_key = "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"

def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def classify_paper_titles(titles, model="gpt-4o-mini"):
    """
    Classify a list of paper titles as AI-related or Not AI-related using OpenAI's GPT API.
    
    :param titles: A list of paper titles.
    :param model: The GPT model to use (default: "gpt-4").
    :return: A list of classifications ("AI-related" or "Not AI-related").
    """
    # Create a single prompt for batch processing
    titles_text = "\n".join([f"{i+1}. {title}" for i, title in enumerate(titles)])
    messages = [
        {"role": "system", "content": "You are an assistant that determines whether paper titles are related to Artificial Intelligence (AI)."},
        {"role": "user", "content": f"""
        Determine if each of the following paper titles is related to Artificial Intelligence (AI). 
        Respond with a list where each title is followed by either 'AI-related' or 'Not AI-related'. 
        
        Titles:
        {titles_text}
        """}
    ]
    
    try:
        # Make the API call
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.0  # Deterministic output for classification
        )
        # Extract and parse the response
        result_text = response.choices[0].message['content'].strip()
        return result_text.split("\n")  # Return each classification as a list
    except Exception as e:
        print(f"Error: {e}")
        return None


def classify_paperx(title, abstract, model="gpt-4"):
    """
    Classify a paper as AI-related or Not AI-related using its title and abstract.
    
    :param title: The title of the paper.
    :param abstract: The abstract of the paper.
    :param model: The GPT model to use (default: "gpt-4").
    :return: "AI-related" or "Not AI-related".
    """
    # Define the classification prompt
    messages = [
        {"role": "system", "content": "You are an expert research assistant in Natural Language Processing (NLP)."},
        {"role": "user", "content": f"""
        Given a paper’s title and abstract, decide whether the paper should be classified as an NLP paper or Not NLP paper.

         A paper should be classified as NLP if it is primarily about:
            - Natural language processing or computational linguistics
            - Speech processing (recognition, synthesis, spoken dialogue)
            - Multimodal tasks involving language (e.g., vision–language, speech–text)
            - Dataset curation, annotation, or benchmarks for NLP/multimodal tasks
            - Core NLP tasks (e.g., text classification, parsing, entity linking, translation, summarization, information extraction, question answering, dialogue systems, sentiment analysis, topic classification, misinformation detection)
            - Information retrieval, data mining or text mining centered on language
            - Evaluation metrics for NLP tasks (e.g., BLEU, ROUGE, COMET, etc.)
            - Language modeling
            - Ethics, bias, fairness, interpretability, safety in NLP systems
            - Human-centered NLP (e.g., user-centered design, human–AI interaction, social impact)
            - Applied NLP in specialized domains (e.g., agriculture, biomedical, legal, educational, social media)

        Output only one label: NLP or Not NLP
        
        Title: "{title}"
        
        Abstract: "{abstract}"
        """}
    ]
    # print(messages)
    
    try:
        # Make the API call
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.0  # Deterministic output for classification
        )
        # Extract and return the classification result
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

def classify_paper(title, abstract, model="gpt-4"):
    """
    Classify a paper as AI-related or Not AI-related using its title and abstract.
    
    :param title: The title of the paper.
    :param abstract: The abstract of the paper.
    :param model: The GPT model to use (default: "gpt-4").
    :return: "AI-related" or "Not AI-related".
    """
    # Define the classification prompt
    messages = [
        {"role": "system", "content": "You are an expert research assistant in Natural Language Processing (NLP)."},
        {"role": "user", "content": f"""
        Given a paper's title and abstract, decide whether the paper should be classified as "relevant" or "not relevant".

        A paper should be classified as relevant if it is about:
            - Natural language processing (NLP) or computational linguistics
            - Speech processing (e.g., recognition, synthesis, spoken dialogue)
            - Multimodal tasks involving language (e.g., vision-language, speech-text, OCR with text processing, audio-text)
            - Dataset curation, annotation, or benchmarks for NLP/multimodal tasks
            - Core NLP tasks (e.g., text classification, parsing, entity linking, translation, summarization, information extraction, question answering, dialogue systems, sentiment analysis, semantics, discourse, topic classification, misinformation detection)
            - Information retrieval, text mining, or data mining involving text or language understanding
            - Evaluation metrics for NLP tasks (e.g., BLEU, ROUGE, COMET, etc.)
            - Language modeling
            - Language generation
            - Ethics, bias, fairness, interpretability, safety in language technology systems
            - Human-centered NLP (e.g., user-centered design of language technology, human-LLM interaction, social impact of language technology)
            - Applied NLP in specialized domains (e.g., agriculture, biomedical, legal, education, cultural analytics, computational social science, NLP for social good)

        Otherwise, the paper should be classified as not relevant.

        Output only one label: "relevant" or "not relevant".

        Title: "{title}"

        Abstract: "{abstract}"
        """}
    ]
    # print(messages)
    
    try:
        # Make the API call
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.0  # Deterministic output for classification
        )
        # Extract and return the classification result
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error: {e}")
        return None
    
# Batch processing function
def classify_papers_in_batches(papers, batch_size=5, model="gpt-4o-2024-08-06"):
    """
    Classify multiple papers in batches using their title and abstract.
    
    :param papers: A list of dictionaries with 'title' and 'abstract'.
    :param batch_size: Number of papers to process in each batch.
    :param model: The GPT model to use (default: "gpt-4").
    :return: A list of classifications.
    """
    classifications = []
    batches = [papers[i:i + batch_size] for i in range(0, len(papers), batch_size)]

    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing Batches")):
        batch_results = []
        # for batch in tqdm([papers[i:i + batch_size] for i in range(0, len(papers), batch_size)], desc="Processing Batches"):
        # batch_results = []
        for paper in batch:
            classification = classify_paper(paper['title'], paper['abstract'], model)
            batch_results.append({
                "title": paper['title'],
                "classification": classification
            })
            # Optional: Sleep to avoid rate limits
            time.sleep(2)
        classifications.extend(batch_results)
        # Sleep after every 500 batches
        if (batch_idx + 1) % 500 == 0:
            print(f"Sleeping after {batch_idx + 1} batches...")
            time.sleep(180)  # sleep for 1 minute (adjust as needed)
    return classifications

def write_result(filename, result):
    with open(filename,'a') as f:
        for item in result:
            print(item)
            f.write(item.strip())
            f.write("\n")

# Example usage
if __name__ == "__main__":
    # List of paper titles to classify
    paper_titles = [
        "Deep Reinforcement Learning for Autonomous Driving",
        "Analyzing the Effect of Soil Composition on Crop Yield",
        "Neural Networks for Image Recognition: A Survey",
        "The Impact of Climate Change on Arctic Ecosystems",
        "Algorithmic Behaviors Across Regions: A Geolocation Audit of YouTube Search for COVID-19 Misinformation between the United States and South Africa"
    ]
    
    category = sys.argv[1]
    print(category)
    # Get classifications for all titles
    classifications = classify_paper_titles(paper_titles)
    # Load the Excel file
    file_path = "/data/users/jalabi/AfricaNLP_Survey/openai/latest/merged_output_final.xlsx"
    excel_data = pd.ExcelFile(file_path)
    sheet_x1 = excel_data.parse(category)  # Sheet named 'X1'
    sheet_x1['title'] = sheet_x1['title'].fillna("")
    sheet_x1['abstract'] = sheet_x1['abstract'].fillna("")
    paper_titles = sheet_x1['title'].tolist()
    paper_abstrat = sheet_x1['abstract'].tolist()

    papers = [{'title':x.strip(),'abstract':y.strip()} for x,y in zip(paper_titles, paper_abstrat)] 

    # Process the papers
    results = classify_papers_in_batches(papers, batch_size=2, model="gpt-4o-2024-08-06")
    if len(results) > 0:
        # write_result(f'{category}.txt', results)
        with open(f'{category}.json', "w") as f:
            json.dump(results, f, indent=4)
