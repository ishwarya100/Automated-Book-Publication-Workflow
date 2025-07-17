import streamlit as st
import asyncio
import os
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag
import random
from playwright.async_api import async_playwright
import textstat
import chromadb
from chromadb.utils import embedding_functions
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import uuid
from datetime import datetime
import traceback

# Set WindowsProactorEventLoopPolicy to fix NotImplementedError
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Ensure NLTK data
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    nltk.data.find('corpora/words')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('words', quiet=True)

# --- Scraper ---
async def preprocess_text_scraper(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text.strip())
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\.\.+', '.', text)
    text = re.sub(r'\s+', ' ', text)
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    return text.strip()

async def scrape_chapter(url, output_dir):
    try:
        print(f"Scraping {url} at {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            print("Browser launched.")
            page = await browser.new_page()
            try:
                await page.goto(url, timeout=60000)
                await page.wait_for_load_state('networkidle', timeout=60000)
                print("Navigation and network idle state achieved.")
            except Exception as e:
                error_msg = f"Navigation failed: {type(e).__name__}: {str(e)}\nTraceback: {traceback.format_exc()}"
                print(error_msg)
                print(f"Debug HTML: {(await page.content())[:1000]}")
                await page.close()
                await browser.close()
                return {"error": error_msg}
            
            content = await page.content()
            if "Wikipedia does not have an article with this exact name" in content:
                error_msg = "No article found at this URL. Consider a different page."
                print(error_msg)
                await page.close()
                await browser.close()
                return {"error": error_msg}
            
            try:
                article_content = await page.query_selector('div#mw-content-text')
                if article_content:
                    text = await article_content.inner_text()
                    print("Extracted article text.")
                else:
                    text = await page.content()
                    print("Fallback to full content.")
            except Exception as e:
                error_msg = f"Content extraction failed: {type(e).__name__}: {str(e)}\nTraceback: {traceback.format_exc()}"
                print(error_msg)
                print(f"Debug HTML: {(await page.content())[:1000]}")
                await page.close()
                await browser.close()
                return {"error": error_msg}
            
            if not text or not text.strip():
                error_msg = "No content extracted. Check URL or page structure."
                print(error_msg)
                print(f"Debug HTML: {(await page.content())[:1000]}")
                await page.close()
                await browser.close()
                return {"error": error_msg}
            
            text = await preprocess_text_scraper(text)
            
            os.makedirs(output_dir, exist_ok=True)
            text_path = os.path.join(output_dir, "chapter_text.txt")
            screenshot_path = os.path.join(output_dir, "chapter_screenshot.png")
            
            try:
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Saved text: {text_path}")
            except Exception as e:
                print(f"Error saving text: {type(e).__name__}: {str(e)}")
            
            try:
                await page.screenshot(path=screenshot_path, full_page=True)
                print(f"Saved screenshot: {screenshot_path}")
            except Exception as e:
                print(f"Error saving screenshot: {type(e).__name__}: {str(e)}")
            
            word_count = len(text.split())
            print(f"Scraped {word_count} words.")
            await page.close()
            await browser.close()
            return {
                "chapter_text": text,
                "screenshot_path": screenshot_path,
                "text_path": text_path,
                "word_count": word_count
            }
    except Exception as e:
        error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_msg)
        return {"error": error_msg}

# --- Writer ---
def get_synonym(word, pos):
    synsets = wordnet.synsets(word, pos=pos)
    if not synsets:
        return word
    synonyms = set(lemma.name().replace('_', ' ') for syn in synsets for lemma in syn.lemmas() if lemma.name().lower() != word.lower())
    return random.choice(list(synonyms)) if synonyms else word

def spin_text(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    spun_tokens = []
    for word, pos in pos_tags:
        if pos.startswith(('NN', 'VB')) and random.random() < 0.2:
            spun_tokens.append(get_synonym(word, pos[0].lower()))
        else:
            spun_tokens.append(word)
    spun_text = "".join(token if token in '.!?,;' else token + " " for token in spun_tokens).strip()
    return spun_text

def rewrite_chapter(input_path, output_path):
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file {input_path} not found.")
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text or len(text.split()) < 5:
            raise ValueError("Input text is empty or too short to rewrite.")
        spun_text = spin_text(text)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(spun_text)
        print(f"Saved rewritten text: {output_path}")
        return {
            "original_text": text,
            "spun_text": spun_text,
            "output_path": output_path,
            "word_count_original": len(text.split()),
            "word_count_spun": len(spun_text.split())
        }
    except Exception as e:
        print(f"Rewrite error: {type(e).__name__}: {str(e)}")
        return None

# --- Reviewer ---
def analyze_text(text):
    try:
        text = re.sub(r'\s+', ' ', text.strip()).rstrip('.') + '.'
        word_count = len(word_tokenize(text))
        sentences = sent_tokenize(text)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        readability_score = textstat.flesch_reading_ease(text)
        suggestions = []
        if readability_score < 60:
            suggestions.append("Text may be too complex (Flesch score < 60).")
        if any(len(word_tokenize(sent)) > 30 for sent in sentences):
            suggestions.append("Some sentences exceed 30 words.")
        if word_count < 100:
            suggestions.append("Text is very short (<100 words).")
        if sentence_count < 5:
            suggestions.append("Text has very few sentences.")
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "readability_score": readability_score,
            "suggestions": suggestions
        }
    except Exception as e:
        print(f"Analyze error: {type(e).__name__}: {str(e)}")
        return None

def review_chapter(input_path, output_path):
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file {input_path} not found.")
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        analysis = analyze_text(text)
        if not analysis:
            raise ValueError("Text analysis failed.")
        report = (f"Review Report\n{'='*30}\n"
                  f"Word Count: {analysis['word_count']}\n"
                  f"Sentence Count: {analysis['sentence_count']}\n"
                  f"Average Sentence Length: {analysis['avg_sentence_length']:.1f} words\n"
                  f"Flesch Reading Ease: {analysis['readability_score']:.1f}\n\nSuggestions:\n"
                  f"{''.join(f'{i}. {s}\n' for i, s in enumerate(analysis['suggestions'], 1)) or 'No issues detected.\n'}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved review: {output_path}")
        return {"analysis": analysis, "report_path": output_path}
    except Exception as e:
        print(f"Review error: {type(e).__name__}: {str(e)}")
        return None

# --- Feedback ---
def preprocess_text(text):
    # Normalize text: remove non-ASCII, extra spaces, and multiple punctuation
    text = re.sub(r'[^\x00-\x7F]+', ' ', text.strip())
    text = re.sub(r'\.\.+', '.', text)
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\.\s*', '. ', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r'\s*:\s*', ': ', text)
    text = re.sub(r'\s*\?\s*', '? ', text)
    # Remove metadata, prompts, and artifacts
    patterns = [
        r'Book of account \d+\..*?Chapter \d+\..*?(?=\.\s+[A-Z])',
        r'The gates of morning\/|The gates of morning \'\'',
        r'``.*?``',
        r'Chapter i\.',
        r'by [a-zA-Z\s]+\.?',
        r'Hand \d+\.',
        r'Bible \d+\.',
        r'\(.*?kamina tai.*?\)',
        r'(?i)Paraphrase:.*?\.',
        r'(?i)Simplify and enhance:.*?\.',
        r'(?i)Einfacher und besser:.*?\.',
        r'(?i)vereinfacht und verbessert.*?\.',
        r'(?i)Simplifier et.*?\.',
        r'False\.|True\.',
        r':\s*:',
        r'\.\s*\.',
        r'\s+\.',
        r',\s*,',
        r'\?\.',
        r'\s*\?\s*\.'
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    # Ensure proper sentence ending
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    return text.strip()

def post_process_text(text):
    # General replacements for common errors
    replacements = [
        (r'\bcast his eyes\b', 'gazed'), (r'\broar\b', 'crashed'), (r'\bspray scattered\b', 'spray faded'),
        (r'\bstretched\b', 'spread'), (r'\bspace\b', 'vast'), (r'\bfishing cod\b', 'fishing birds'),
        (r'\bsportfishing\b', 'fishing'), (r'\bnip by\b', 'surrounded by'), (r'\bring\b', 'encircle'),
        (r'\bgreat pool\b', 'mighty sea'), (r'\bsea in itself\b', 'sea itself'), (r'\bget landed\b', 'arrived'),
        (r'\bwas learn\b', 'seen'), (r'\bcomprise\b', 'were'), (r'\bsign of the zodiac\b', 'houses'),
        (r'\blagune\b', 'lagoon'), (r'\bbrow contracted\b', 'brow furrowed'), 
        (r'\blife story\b', 'life'), (r'\btike\b', 'children'), (r'\bsauceboat\b', 'boat'),
        (r'\bembody\b', 'been'), (r'\bsupreme being\b', 'gods'), (r'\bpublic figure\b', 'name'),
        (r'\bmake up\b', 'lived'), (r'\bin habit\b', 'inhabited'), (r'\bthe the\b', 'the'),
        (r'\bof of\b', 'of'), (r'\bca-ca\b', 'made'), (r'\btake a shit\b', 'make'),
        (r'\bgraven image\b', 'gods'), (r'\bpersonify\b', 'was'), (r'\bmalarky\b', 'storm'),
        (r'\bshaver\b', 'child'), (r'\bfry\b', 'children'), (r'\bdominicus\b', 'sun'),
        (r'\bstate of war\b', 'war'), (r'\bpiddle\b', 'water'), (r'\bgeorge sand\b', 'sand'),
        (r'\bvommit\b', 'from'), (r'\bblook\b', 'savage'), (r'\bkanaka\b', 'native'),
        (r'\bpullulate\b', 'gathered'), (r'\boculus\b', 'eyes'), (r'\bwere was\b', 'was'),
        (r'\bmy is\b', 'my name is'), (r'\bwas his he who\b', 'he'), (r'\bbang that all the world of swept away by had left\b', 'knew war had swept away'),
        (r'\bwho cost you\b', 'who are you'), (r'\bone day long ago arrive\b', 'one day long ago arrived'),
        (r'\bwere blow out\b', 'was blown out'), (r'\btari tari\b', 'Tari'),
        (r'\bi need as i breathe but\b', 'I ask because I must'), (r'\bno big folk madyana will ever answer\b', 'no one will ever answer'),
        (r'\bturn over his gaze\b', 'turned his gaze'), (r'\bhad submitted possession of\b', 'had taken'),
        (r'\bget a line the far mirror blaze of\b', 'see the distant glow of'), (r'\bdick were an all but\b', 'Dick was nearly a'),
        (r'\bthe white man were there\b', 'he, the white man, was there'), (r'\bcanoe! did he intend fighting\b', 'Canoes! Did he plan to fight'),
        (r'\bdid he vaguely intend to be the attacker\b', 'did he plan to attack'), (r'\bat big intervals and in\b', 'at times'),
        (r'\bthough far behind the sea course\b', 'far across the sea'), (r'\bwas visible from through a\b', 'was visible through a mirage'),
        (r'\bnow again it were beginning to live\b', 'now it began to glow'), (r'\btemper till as if sketched in by some unseen painter\b', 'until it appeared, drawn by an unseen hand'),
        (r'\btheir minds untutored, knowing\b', 'their simple minds knew'), (r'\bnothing of, their\b', 'nothing of their past'),
        (r'\bthere was no world beyond the water encircling the islands\b', 'no world existed beyond the encircling waters'),
        (r'\ba hand fell upon his shoulder and\.,\b', 'a hand touched his shoulder, and'), (r'\bi make out not know\b', 'I do not know')
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Remove non-English words
    english_words = set(nltk.corpus.words.words())
    text = re.sub(r'\b[a-zA-ZàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]+\b', lambda m: '' if m.group(0).lower() not in english_words else m.group(0), text)
    # Fix punctuation
    text = re.sub(r'\.\s*,\s*', '.', text)
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r'\.\.+', '.', text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r'\?\.', '?', text)
    text = re.sub(r'\s*\?\s*\.', '?', text)
    text = re.sub(r'\s*\.\s*$', '.', text)
    text = re.sub(r'\s*\?\s*$', '?', text)
    text = re.sub(r'\s*[,.!?:;]\s*[,.!?:;]', '.', text)
    return text.strip()

def split_long_sentence(sentence):
    tokens = word_tokenize(sentence)
    if len(tokens) <= 20:
        return [sentence.strip()]
    
    split_sentences = []
    current_sentence = []
    word_count = 0
    break_points = {',', ';', 'and', 'but', 'or', '—', ':', 'while', 'as', 'before', 'after', '-', '(', ')'}
    
    for i, token in enumerate(tokens):
        current_sentence.append(token)
        word_count += 1
        if (token in break_points or (i > 0 and tokens[i-1] in break_points)) and word_count >= 5:
            split_sentences.append(' '.join(current_sentence).strip() + '.')
            current_sentence = []
            word_count = 0
    if current_sentence:
        split_sentences.append(' '.join(current_sentence).strip() + '.')
    
    final_sentences = []
    for sent in split_sentences:
        sent_tokens = word_tokenize(sent)
        while len(sent_tokens) > 20:
            mid = len(sent_tokens) // 2
            final_sentences.append(' '.join(sent_tokens[:mid]).strip() + '.')
            sent_tokens = sent_tokens[mid:]
        final_sentences.append(' '.join(sent_tokens).strip() + '.')
    
    return [s for s in final_sentences if s.strip() and not s.lower().startswith(('paraphrase', 'simplify', 'book', 'chapter', 'einfacher', 'vereinfacht', 'simplifier'))]

def ai_improve_story(text, model, tokenizer, device, batch_size=4):
    try:
        processed_text = preprocess_text(text)
        sentences = sent_tokenize(processed_text)
        if not sentences:
            return text
        
        improved_sentences = []
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            split_batch = []
            for sent in batch_sentences:
                split_batch.extend(split_long_sentence(sent))
            
            split_batch = [s for s in split_batch if len(word_tokenize(s)) >= 3 and not any(x in s.lower() for x in ['book', 'chapter', 'paraphrase', 'simplify', 'einfacher', 'vereinfacht', 'simplifier'])]
            if not split_batch:
                continue
            
            input_texts = [f"simplify and enhance to clear English: {sent}" for sent in split_batch]
            inputs = tokenizer(input_texts, return_tensors='pt', max_length=512, truncation=True, padding=True).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=4,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            
            for output in outputs:
                rewritten = tokenizer.decode(output, skip_special_tokens=True)
                rewritten = post_process_text(rewritten)
                tokens = word_tokenize(rewritten)
                if len(tokens) <= 20 and len(tokens) >= 3 and rewritten.strip() and all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.!? ' for c in rewritten):
                    improved_sentences.append(rewritten.strip().capitalize() + '.')
        
        improved_text = ' '.join(improved_sentences)
        improved_text = re.sub(r'\s*\.\s*\.', '.', improved_text)
        improved_text = re.sub(r'\s+', ' ', improved_text)
        improved_text = re.sub(r'\s*,\s*', ', ', improved_text)
        improved_text = re.sub(r'\s*\.\s*$', '.', improved_text)
        improved_text = re.sub(r'\s*\?\s*$', '?', improved_text)
        return improved_text.strip() or text
    except Exception as e:
        print(f"Error improving story: {str(e)}")
        return text

def read_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None

def save_feedback(feedback, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(feedback)
        print(f"Saved feedback to {output_path}: {os.path.exists(output_path)}")
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")

def collect_feedback(rewritten_path, report_path, feedback_path, feedback_type):
    try:
        rewritten_text = read_file(rewritten_path)
        if not rewritten_text:
            raise FileNotFoundError(f"Rewritten chapter {rewritten_path} not found.")
        
        report_text = read_file(report_path)
        if not report_text:
            raise FileNotFoundError(f"Review report {report_path} not found.")
        
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        if feedback_type == "2":
            improved_story = ai_improve_story(rewritten_text, model, tokenizer, device, batch_size=4)
            feedback = f"AI-Improved Story:\n{improved_story}"
            save_feedback(feedback, feedback_path)
            return {"feedback": feedback, "feedback_path": feedback_path}
        else:
            print("Feedback type 1 not supported in this integrated mode.")
            return None
    
    except Exception as e:
        print(f"Error during feedback collection: {str(e)}")
        return None

# --- Store ---
def store_in_chromadb(file_path, collection_name, story_id, story_type, db_dir):
    try:
        text = read_file(file_path)
        if not text:
            raise FileNotFoundError(f"File {file_path} not found.")
        sentences = sent_tokenize(text)
        if not sentences:
            raise ValueError("No sentences found.")
        client = chromadb.PersistentClient(path=db_dir)
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        )
        ids = [f"{story_id}_{i}" for i in range(len(sentences))]
        metadatas = [{"story_id": story_id, "story_type": story_type, "sentence_index": i, "timestamp": datetime.now().isoformat()} for i in range(len(sentences))]
        collection.add(ids=ids, documents=sentences, metadatas=metadatas)
        print(f"Stored {len(sentences)} sentences ({story_type}) in ChromaDB.")
        return {"story_id": story_id, "num_sentences": len(sentences), "story_type": story_type}
    except Exception as e:
        print(f"Store error: {type(e).__name__}: {str(e)}")
        return None

# --- RL Search ---
def reward_function(text):
    try:
        sentences = sent_tokenize(text)
        if not sentences:
            return 0
        score = sum(10 if len(word_tokenize(sent)) <= 20 else -2 for sent in sentences)
        return min(max((score / (len(sentences) * 10)) * 100, 0), 100)
    except Exception:
        return 0

def rl_search(collection_name, db_dir):
    try:
        client = chromadb.PersistentClient(path=db_dir)
        collection = client.get_collection(name=collection_name)
        story_types = ["original", "rewritten", "feedback"]
        best_type, best_score = None, -float('inf')
        for story_type in story_types:
            results = collection.get(where={"story_type": story_type}, include=["documents", "metadatas"])
            if results['documents']:
                text = " ".join(results['documents'])
                score = reward_function(text)
                if score > best_score:
                    best_score = score
                    best_type = story_type
        if best_type:
            print(f"Best version: {best_type} ({int(best_score)}%)")
            return {"story_type": best_type, "score": f"{int(best_score)}%"}
        return None
    except Exception as e:
        print(f"RL search error: {type(e).__name__}: {str(e)}")
        return None

# --- Main Pipeline ---
async def run_pipeline(url, output_dir, db_dir, feedback_type):
    try:
        version_id = str(uuid.uuid4())
        print(f"Pipeline started (Version ID: {version_id}) at {datetime.now().strftime('%I:%M %p IST, %B %d, %Y')}")
        
        scrape_result = await scrape_chapter(url, output_dir)
        if "error" in scrape_result:
            return {"error": scrape_result["error"]}
        
        original_path = os.path.join(output_dir, "chapter_text.txt")
        rewritten_path = os.path.join(output_dir, "rewritten_chapter.txt")
        review_path = os.path.join(output_dir, "review_report.txt")
        feedback_path = os.path.join(output_dir, "feedback.txt")
        
        rewrite_result = rewrite_chapter(original_path, rewritten_path)
        if not rewrite_result:
            return {"error": "Rewrite failed."}
        
        review_result = review_chapter(rewritten_path, review_path)
        if not review_result:
            return {"error": "Review failed."}
        
        feedback_result = collect_feedback(rewritten_path, review_path, feedback_path, feedback_type)
        
        collection_name = "story_collection"
        storage_results = []
        for file_path, story_type in [(original_path, "original"), (rewritten_path, "rewritten"), (feedback_path, "feedback")]:
            if os.path.exists(file_path):
                store_result = store_in_chromadb(file_path, collection_name, version_id, story_type, db_dir)
                if store_result:
                    storage_results.append(store_result)
        
        rl_result = rl_search(collection_name, db_dir)
        
        return {
            "version_id": version_id,
            "scrape_result": scrape_result,
            "rewrite_result": rewrite_result,
            "review_result": review_result,
            "feedback_result": feedback_result,
            "storage_results": storage_results,
            "rl_result": rl_result
        }
    except Exception as e:
        print(f"Pipeline error: {type(e).__name__}: {str(e)}\nTraceback: {traceback.format_exc()}")
        return {"error": f"Pipeline failed: {type(e).__name__}: {str(e)}"}

# --- Streamlit UI ---
st.title("Book Publication Workflow")

# Allow any URL with a default value
url = st.text_input(
    "Enter URL of Book Chapter:",
    value="https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
)

# Require user to input output directory path (no default value)
output_dir = st.text_input(
    "Enter Output Directory Path (example: C:/Users/user/Desktop/output):"
)

# Require user to input ChromaDB directory path (no default value)
db_dir = st.text_input(
    "Enter ChromaDB Directory Path (example: C:/Users/user/Desktop/chroma_db):"
)

feedback_type = st.radio(
    "Feedback type:", 
    ["AI-improved story"]
)

if st.button("Run Pipeline"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)
    with st.spinner("Running..."):
        try:
            result = asyncio.run(run_pipeline(url, output_dir, db_dir, feedback_type[0]))
            st.session_state["result"] = result
        except Exception as e:
            st.error(f"Pipeline error: {type(e).__name__}: {str(e)}")
            result = {"error": f"Pipeline failed: {type(e).__name__}: {str(e)}"}
    
    st.subheader("Output")
    if "error" in result:
        st.error(result["error"])
    else:
        st.write(f"Version ID: {result['version_id']}")
        if result["scrape_result"]:
            st.write(f"Scraped {result['scrape_result']['word_count']} words to {result['scrape_result']['text_path']}")
        if result["rewrite_result"]:
            st.write(f"Rewritten to {result['rewrite_result']['output_path']}")
        if result["review_result"]:
            st.write(f"Review saved to {result['review_result']['report_path']}")
        if result["feedback_result"]:
            st.write(f"Feedback saved to {result['feedback_result']['feedback_path']}")
        if result["storage_results"]:
            for store_result in result["storage_results"]:
                st.write(f"Stored {store_result['num_sentences']} sentences ({store_result['story_type']}) in ChromaDB")
        if result["rl_result"]:
            st.write(f"Best Version: {result['rl_result']['story_type']} ({result['rl_result']['score']})")
        
        st.subheader("Content")
        for file_name, title in [("chapter_text.txt", "Original"), ("rewritten_chapter.txt", "Rewritten"), ("review_report.txt", "Review"), ("feedback.txt", "Feedback")]:
            file_path = os.path.join(output_dir, file_name)
            if os.path.exists(file_path):
                content = read_file(file_path)
                if content:
                    st.write(f"**{title}**:")
                    st.text_area("", content, key=file_name, height=200)
