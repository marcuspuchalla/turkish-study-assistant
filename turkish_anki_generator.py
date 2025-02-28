#!/usr/bin/env python3

import os
import sys
import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import logging

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF not installed. Install with: pip install pymupdf")
    print("This library is required for PDF text extraction.")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("Error: Anthropic Python SDK not installed. Install with: pip install anthropic")
    print("This library is required for translation with Claude.")
    sys.exit(1)

try:
    import genanki
except ImportError:
    print("Error: genanki not installed. Install with: pip install genanki")
    print("This library is required for creating Anki cards.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_ID = 1607392319  # Random model ID for Anki
DECK_ID = 2059400110  # Random deck ID for Anki

def is_turkish_word(word: str) -> bool:
    """
    Check if a word is likely Turkish by looking for Turkish-specific characters.
    This is a basic heuristic and not 100% accurate.
    """
    turkish_chars = "çğıöşüÇĞİÖŞÜ"
    # Check if any Turkish-specific characters are in the word
    if any(char in word for char in turkish_chars):
        return True
    
    # Check for common Turkish endings (very simple check)
    common_endings = ["lar", "ler", "lık", "lik", "mak", "mek", "ci", "çi", "cı", "çı"]
    if any(word.endswith(ending) for ending in common_endings):
        return True
    
    # Words must be at least 3 characters to be considered
    return len(word) >= 3 and word.isalpha()

def clean_word(word: str) -> str:
    """Clean up a word by removing punctuation and converting to lowercase."""
    word = re.sub(r'[^\w\sçğıöşüÇĞİÖŞÜ]', '', word)
    return word.strip().lower()

def extract_turkish_words(pdf_path: str) -> Set[str]:
    """Extract what appear to be Turkish words from a PDF."""
    doc = fitz.open(pdf_path)
    all_words = set()
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        
        # Replace multiple spaces and newlines with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Split text into words
        words = text.split()
        for word in words:
            cleaned = clean_word(word)
            if cleaned and is_turkish_word(cleaned):
                all_words.add(cleaned)
    
    doc.close()
    return all_words

def translate_turkish_words(words: List[str], api_key: str, batch_size: int = 20) -> Dict[str, Dict]:
    """
    Translate Turkish words to English using Claude.
    Returns a dictionary of {turkish_word: {translation, example}}
    """
    client = anthropic.Anthropic(api_key=api_key)
    translations = {}
    
    # Process in batches to avoid rate limits and long contexts
    for i in range(0, len(words), batch_size):
        batch = words[i:i+batch_size]
        logger.info(f"Translating batch {i//batch_size + 1} ({len(batch)} words)")
        
        # Create the prompt for translating a batch of words
        words_list = "\n".join([f"- {word}" for word in batch])
        prompt = f"""
I have a list of Turkish words that I need translated to English. For each word, please provide:
1. The English translation
2. An example sentence using the Turkish word
3. The English translation of that sentence

Please format your response as a JSON array of objects, with each object having these fields:
- turkish_word: The original Turkish word
- english_translation: The English translation of the word
- example_sentence: A simple example sentence in Turkish
- example_translation: English translation of the example sentence

Here are the words:
{words_list}
        """
        
        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                temperature=0,
                system="You are a Turkish language expert, fluent in both Turkish and English. Provide accurate translations and examples for Turkish words.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract JSON from response
            content = response.content[0].text
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    logger.warning(f"Could not extract JSON from response for batch {i//batch_size + 1}")
                    logger.debug(f"Response content: {content}")
                    continue
            
            # Load the JSON response
            try:
                translated_data = json.loads(json_str)
                
                # Add to our translations dictionary
                for entry in translated_data:
                    turkish_word = entry.get("turkish_word", "").strip()
                    if turkish_word:
                        translations[turkish_word] = {
                            "english": entry.get("english_translation", ""),
                            "example": entry.get("example_sentence", ""),
                            "example_translation": entry.get("example_translation", "")
                        }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from response: {e}")
                logger.debug(f"Content causing the error: {json_str}")
        
        except Exception as e:
            logger.error(f"Error with Claude API: {e}")
        
        # Sleep a bit to avoid hitting rate limits
        if i + batch_size < len(words):
            time.sleep(2)
    
    return translations

def create_anki_deck(translations: Dict[str, Dict], output_dir: str, deck_name: str = "Turkish Vocabulary"):
    """Create Anki cards from the translations."""
    # Define the model (card template)
    model = genanki.Model(
        MODEL_ID,
        'Turkish Vocabulary Model',
        fields=[
            {'name': 'Turkish'},
            {'name': 'English'},
            {'name': 'Example'},
            {'name': 'ExampleTranslation'},
        ],
        templates=[
            {
                'name': 'Turkish to English',
                'qfmt': '<h1>{{Turkish}}</h1>',
                'afmt': '<h1>{{Turkish}}</h1><hr id="answer"><h2>{{English}}</h2><p><i>{{Example}}</i><br>{{ExampleTranslation}}</p>',
            },
            {
                'name': 'English to Turkish',
                'qfmt': '<h1>{{English}}</h1>',
                'afmt': '<h1>{{English}}</h1><hr id="answer"><h2>{{Turkish}}</h2><p><i>{{Example}}</i><br>{{ExampleTranslation}}</p>',
            },
        ]
    )
    
    # Create a deck
    deck = genanki.Deck(DECK_ID, deck_name)
    
    # Add notes (cards) to the deck
    for turkish_word, data in translations.items():
        note = genanki.Note(
            model=model,
            fields=[
                turkish_word,
                data.get('english', ''),
                data.get('example', ''),
                data.get('example_translation', '')
            ]
        )
        deck.add_note(note)
    
    # Create output directory if it doesn't exist
    anki_dir = os.path.join(output_dir, 'anki')
    os.makedirs(anki_dir, exist_ok=True)
    
    # Save the deck
    deck_path = os.path.join(anki_dir, f"{deck_name.replace(' ', '_')}.apkg")
    genanki.Package(deck).write_to_file(deck_path)
    
    # Also save as JSON for easier inspection
    json_path = os.path.join(anki_dir, f"{deck_name.replace(' ', '_')}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(translations, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Created Anki deck with {len(translations)} cards at {deck_path}")
    logger.info(f"Saved translations as JSON at {json_path}")
    
    return deck_path

def process_pdfs(input_path: str, output_dir: str, api_key: str):
    """Process all PDFs in the input path."""
    pdf_paths = []
    
    # Check if input_path is a directory or a file
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_paths.append(os.path.join(root, file))
    elif os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
        pdf_paths.append(input_path)
    elif input_path.endswith('.txt'):
        # Assume it's a file with a list of PDF paths
        with open(input_path, 'r') as f:
            pdf_paths = [line.strip() for line in f if line.strip().lower().endswith('.pdf')]
    else:
        logger.error(f"Invalid input: {input_path}. Please provide a PDF file, a directory containing PDFs, or a text file listing PDF paths.")
        return
    
    if not pdf_paths:
        logger.error("No PDF files found.")
        return
    
    logger.info(f"Found {len(pdf_paths)} PDF files to process")
    
    # Extract words from all PDFs
    all_turkish_words = set()
    for pdf_path in pdf_paths:
        logger.info(f"Extracting words from {pdf_path}")
        try:
            words = extract_turkish_words(pdf_path)
            all_turkish_words.update(words)
            logger.info(f"Extracted {len(words)} potential Turkish words")
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
    
    logger.info(f"Total unique Turkish words found: {len(all_turkish_words)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the list of words
    words_list_path = os.path.join(output_dir, 'turkish_words.txt')
    with open(words_list_path, 'w', encoding='utf-8') as f:
        for word in sorted(all_turkish_words):
            f.write(f"{word}\n")
    
    logger.info(f"Saved word list to {words_list_path}")
    
    # Translate words
    word_list = list(all_turkish_words)
    translations = translate_turkish_words(word_list, api_key)
    
    # Create Anki cards
    if translations:
        create_anki_deck(translations, output_dir)
    else:
        logger.warning("No translations were generated, cannot create Anki deck")

def main():
    parser = argparse.ArgumentParser(description='Generate Anki cards from Turkish PDFs')
    parser.add_argument('input', help='PDF file, directory containing PDFs, or a text file with PDF paths')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--api-key', '-k', help='Anthropic API key')
    parser.add_argument('--batch-size', '-b', type=int, default=20, help='Batch size for translation requests')
    
    args = parser.parse_args()
    
    # Check for API key in environment if not provided as argument
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("Anthropic API key must be provided either as an argument or in the ANTHROPIC_API_KEY environment variable")
        return
    
    process_pdfs(args.input, args.output, api_key)

if __name__ == "__main__":
    main()