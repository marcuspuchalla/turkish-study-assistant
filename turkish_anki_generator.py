#!/usr/bin/env python3

import os
import sys
import argparse
import json
import re
import time
import signal
import pickle
import shutil
import zipfile
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any, Union
import logging
from datetime import datetime

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

# Try to import rich for better terminal UI
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.prompt import Confirm
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: For a better UI, install rich: pip install rich")

# Configure logging
if RICH_AVAILABLE:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    console = Console()
else:
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
STATE_FILE = "turkish_study_state.pickle"  # File to save progress
COLLECTION_INFO_FILE = "collection_info.json"  # File to store collection metadata
PROCESSED_PDFS_FILE = "processed_pdfs.json"  # File to track processed PDFs
DEFAULT_COLLECTION = "turkish_vocabulary"  # Default collection name

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
    """
    Extract what appear to be Turkish words from a PDF.
    Uses a progress bar when rich library is available.
    """
    doc = fitz.open(pdf_path)
    all_words = set()
    total_pages = len(doc)
    
    if RICH_AVAILABLE and total_pages > 1:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            page_task = progress.add_task(f"[cyan]Reading PDF: {os.path.basename(pdf_path)}[/cyan]", total=total_pages)
            
            for page_num in range(total_pages):
                progress.update(page_task, description=f"[cyan]Reading page {page_num+1}/{total_pages}[/cyan]")
                
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Replace multiple spaces and newlines with a single space
                text = re.sub(r'\s+', ' ', text)
                
                # Split text into words
                words = text.split()
                turkish_words_on_page = 0
                
                for word in words:
                    cleaned = clean_word(word)
                    if cleaned and is_turkish_word(cleaned):
                        all_words.add(cleaned)
                        turkish_words_on_page += 1
                
                progress.update(page_task, advance=1)
                if turkish_words_on_page > 0:
                    progress.print(f"Found {turkish_words_on_page} Turkish words on page {page_num+1}")
    else:
        for page_num in range(total_pages):
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

def translate_turkish_words(words: List[str], api_key: str, batch_size: int = 20, state: Optional['ProcessingState'] = None) -> Dict[str, Dict]:
    """
    Translate Turkish words to English using Claude.
    Returns a dictionary of {turkish_word: {translation, example}}
    
    If a state object is provided, it will be used to save progress periodically.
    """
    client = anthropic.Anthropic(api_key=api_key)
    translations = {}
    
    # Progress tracking 
    total_batches = (len(words) + batch_size - 1) // batch_size
    
    if RICH_AVAILABLE:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            translation_task = progress.add_task("[cyan]Translating words...", total=len(words))
            
            # Process in batches to avoid rate limits and long contexts
            for i in range(0, len(words), batch_size):
                batch = words[i:i+batch_size]
                batch_num = i//batch_size + 1
                
                progress.update(translation_task, 
                               description=f"[cyan]Translating batch {batch_num}/{total_batches} ({len(batch)} words)[/cyan]")
                
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
                    progress.update(translation_task, description=f"[cyan]Making API call for batch {batch_num}/{total_batches}[/cyan]")
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
                    progress.update(translation_task, description=f"[cyan]Processing API response for batch {batch_num}/{total_batches}[/cyan]")
                    
                    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                    
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Try to find JSON without code blocks
                        json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                        else:
                            progress.print(f"[yellow]Could not extract JSON from response for batch {batch_num}/{total_batches}[/yellow]")
                            if logging.getLogger().level <= logging.DEBUG:
                                progress.print(f"[dim]Response content: {content}[/dim]")
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
                                
                        # Update progress
                        progress.update(translation_task, advance=len(batch))
                        
                        # Display a few translations as examples
                        if translated_data and len(translated_data) > 0:
                            sample = translated_data[0]
                            progress.print(
                                f"[green]Sample translation:[/green] "
                                f"[bold]{sample.get('turkish_word', '')}[/bold] → "
                                f"[bold]{sample.get('english_translation', '')}[/bold]"
                            )
                    
                    except json.JSONDecodeError as e:
                        progress.print(f"[red]Failed to parse JSON from response: {e}[/red]")
                        if logging.getLogger().level <= logging.DEBUG:
                            progress.print(f"[dim]Content causing the error: {json_str}[/dim]")
                
                except Exception as e:
                    progress.print(f"[red]Error with Claude API: {e}[/red]")
                
                # Save state after each batch if a state object is provided
                if state:
                    state.translations.update(translations)
                    state.save()
                
                # Sleep a bit to avoid hitting rate limits
                if i + batch_size < len(words):
                    seconds_to_wait = 2
                    progress.update(translation_task, description=f"[cyan]Waiting {seconds_to_wait}s before next batch...[/cyan]")
                    time.sleep(seconds_to_wait)
    else:
        # Process in batches to avoid rate limits and long contexts
        for i in range(0, len(words), batch_size):
            batch = words[i:i+batch_size]
            batch_num = i//batch_size + 1
            logger.info(f"Translating batch {batch_num}/{total_batches} ({len(batch)} words)")
            
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
                logger.info(f"Making API call for batch {batch_num}/{total_batches}")
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
                logger.info(f"Processing API response for batch {batch_num}/{total_batches}")
                
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find JSON without code blocks
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                    else:
                        logger.warning(f"Could not extract JSON from response for batch {batch_num}/{total_batches}")
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
                    
                    # Display a sample translation
                    if translated_data and len(translated_data) > 0:
                        sample = translated_data[0]
                        logger.info(
                            f"Sample translation: "
                            f"{sample.get('turkish_word', '')} → "
                            f"{sample.get('english_translation', '')}"
                        )
                
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from response: {e}")
                    logger.debug(f"Content causing the error: {json_str}")
            
            except Exception as e:
                logger.error(f"Error with Claude API: {e}")
            
            # Save state after each batch if a state object is provided
            if state:
                state.translations.update(translations)
                state.save()
            
            # Sleep a bit to avoid hitting rate limits
            if i + batch_size < len(words):
                seconds_to_wait = 2
                logger.info(f"Waiting {seconds_to_wait}s before next batch...")
                time.sleep(seconds_to_wait)
    
    return translations

def create_anki_deck(translations: Dict[str, Dict], output_dir: str, deck_name: str = "Turkish Vocabulary"):
    """Create Anki cards from the translations."""
    if RICH_AVAILABLE:
        console.print(f"[blue]Creating Anki deck with {len(translations)} cards...[/blue]")
    else:
        logger.info(f"Creating Anki deck with {len(translations)} cards...")
    
    # Define the model (card template)
    model = genanki.Model(
        MODEL_ID,
        'Turkish Vocabulary Model',
        fields=[
            {'name': 'Turkish'},
            {'name': 'English'},
            {'name': 'Example'},
            {'name': 'ExampleTranslation'},
            {'name': 'Notes'},
        ],
        templates=[
            {
                'name': 'Turkish to English',
                'qfmt': '''
<div style="font-family: Arial; font-size: 20px; text-align: center; color: #2c3e50;">
  <h1 style="color: #e74c3c;">{{Turkish}}</h1>
</div>
''',
                'afmt': '''
<div style="font-family: Arial; font-size: 20px; text-align: center; color: #2c3e50;">
  <h1 style="color: #e74c3c;">{{Turkish}}</h1>
  <hr id="answer">
  <h2 style="color: #3498db;">{{English}}</h2>
  <div style="text-align: left; font-size: 18px; margin-top: 20px;">
    <p style="font-style: italic; color: #7f8c8d;">{{Example}}</p>
    <p>{{ExampleTranslation}}</p>
    <p style="font-size: 14px; color: #95a5a6;">{{Notes}}</p>
  </div>
</div>
''',
            },
            {
                'name': 'English to Turkish',
                'qfmt': '''
<div style="font-family: Arial; font-size: 20px; text-align: center; color: #2c3e50;">
  <h1 style="color: #3498db;">{{English}}</h1>
</div>
''',
                'afmt': '''
<div style="font-family: Arial; font-size: 20px; text-align: center; color: #2c3e50;">
  <h1 style="color: #3498db;">{{English}}</h1>
  <hr id="answer">
  <h2 style="color: #e74c3c;">{{Turkish}}</h2>
  <div style="text-align: left; font-size: 18px; margin-top: 20px;">
    <p style="font-style: italic; color: #7f8c8d;">{{Example}}</p>
    <p>{{ExampleTranslation}}</p>
    <p style="font-size: 14px; color: #95a5a6;">{{Notes}}</p>
  </div>
</div>
''',
            },
        ]
    )
    
    # Create a deck
    deck = genanki.Deck(DECK_ID, deck_name)
    
    # Add notes (cards) to the deck
    cards_added = 0
    
    # Use progress bar when rich is available
    if RICH_AVAILABLE and translations:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Creating flashcards...[/cyan]", total=len(translations))
            
            for turkish_word, data in translations.items():
                # Skip words that don't have proper translations
                if not data.get('english') or not data.get('example'):
                    progress.print(f"[yellow]Skipping incomplete card for: {turkish_word}[/yellow]")
                    continue
                
                note = genanki.Note(
                    model=model,
                    fields=[
                        turkish_word,
                        data.get('english', ''),
                        data.get('example', ''),
                        data.get('example_translation', ''),
                        ''  # Empty notes field for user to fill in
                    ]
                )
                deck.add_note(note)
                cards_added += 1
                
                # Show occasional preview
                if cards_added % 100 == 0:
                    progress.print(f"[green]Added {cards_added} cards so far[/green]")
                
                progress.update(task, advance=1)
    else:
        for turkish_word, data in translations.items():
            # Skip words that don't have proper translations
            if not data.get('english') or not data.get('example'):
                logger.warning(f"Skipping incomplete card for: {turkish_word}")
                continue
                
            note = genanki.Note(
                model=model,
                fields=[
                    turkish_word,
                    data.get('english', ''),
                    data.get('example', ''),
                    data.get('example_translation', ''),
                    ''  # Empty notes field for user to fill in
                ]
            )
            deck.add_note(note)
            cards_added += 1
    
    # Create output directory if it doesn't exist
    anki_dir = os.path.join(output_dir, 'anki')
    os.makedirs(anki_dir, exist_ok=True)
    
    # Get timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the deck
    deck_filename = f"{deck_name.replace(' ', '_')}_{timestamp}"
    deck_path = os.path.join(anki_dir, f"{deck_filename}.apkg")
    
    if RICH_AVAILABLE:
        console.print(f"[green]Writing Anki deck to {deck_path}[/green]")
    
    genanki.Package(deck).write_to_file(deck_path)
    
    # Also save as JSON for easier inspection
    json_path = os.path.join(anki_dir, f"{deck_filename}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(translations, f, ensure_ascii=False, indent=2)
    
    if RICH_AVAILABLE:
        console.print(f"[green]Created Anki deck with {cards_added} cards at {deck_path}[/green]")
        console.print(f"[green]Saved translations as JSON at {json_path}[/green]")
    else:
        logger.info(f"Created Anki deck with {cards_added} cards at {deck_path}")
        logger.info(f"Saved translations as JSON at {json_path}")
    
    return deck_path

class CollectionManager:
    """Manages collections of Turkish vocabulary from different sources or levels."""
    
    def __init__(self, base_dir: str):
        """Initialize the collection manager with a base directory."""
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.collections_info = self._load_collections_info()
    
    def _get_collection_info_path(self) -> str:
        """Get the path to the collections info file."""
        return os.path.join(self.base_dir, COLLECTION_INFO_FILE)
    
    def _load_collections_info(self) -> Dict:
        """Load existing collections info or create a new one."""
        info_path = self._get_collection_info_path()
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                if RICH_AVAILABLE:
                    console.print("[yellow]Warning: Could not load collections info file. Creating a new one.[/yellow]")
                else:
                    logger.warning("Could not load collections info file. Creating a new one.")
        
        # Initialize with default structure
        return {
            "collections": {},
            "last_updated": datetime.now().isoformat()
        }
    
    def save_collections_info(self):
        """Save the collections info to the file."""
        self.collections_info["last_updated"] = datetime.now().isoformat()
        with open(self._get_collection_info_path(), 'w', encoding='utf-8') as f:
            json.dump(self.collections_info, f, ensure_ascii=False, indent=2)
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        return collection_name in self.collections_info.get("collections", {})
    
    def get_collection_dir(self, collection_name: str) -> str:
        """Get the directory path for a collection."""
        return os.path.join(self.base_dir, collection_name)
    
    def create_or_update_collection(self, collection_name: str, description: str = "") -> str:
        """Create a new collection or update its metadata."""
        if not collection_name:
            collection_name = DEFAULT_COLLECTION
        
        collection_dir = self.get_collection_dir(collection_name)
        os.makedirs(collection_dir, exist_ok=True)
        
        # Create or update collection info
        if collection_name not in self.collections_info.get("collections", {}):
            self.collections_info.setdefault("collections", {})[collection_name] = {
                "name": collection_name,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "word_count": 0,
                "pdf_count": 0
            }
        else:
            self.collections_info["collections"][collection_name]["last_updated"] = datetime.now().isoformat()
            if description:
                self.collections_info["collections"][collection_name]["description"] = description
        
        self.save_collections_info()
        return collection_dir
    
    def update_collection_stats(self, collection_name: str, word_count: int, pdf_count: int):
        """Update the statistics for a collection."""
        if collection_name in self.collections_info.get("collections", {}):
            self.collections_info["collections"][collection_name]["word_count"] = word_count
            self.collections_info["collections"][collection_name]["pdf_count"] = pdf_count
            self.collections_info["collections"][collection_name]["last_updated"] = datetime.now().isoformat()
            self.save_collections_info()
    
    def list_collections(self) -> List[Dict]:
        """List all available collections with their stats."""
        return list(self.collections_info.get("collections", {}).values())
    
    def get_processed_pdfs(self, collection_name: str) -> Dict[str, Dict]:
        """Get the list of processed PDFs for a collection."""
        pdfs_file = os.path.join(self.get_collection_dir(collection_name), PROCESSED_PDFS_FILE)
        if os.path.exists(pdfs_file):
            try:
                with open(pdfs_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def save_processed_pdfs(self, collection_name: str, pdfs_info: Dict[str, Dict]):
        """Save the list of processed PDFs for a collection."""
        pdfs_file = os.path.join(self.get_collection_dir(collection_name), PROCESSED_PDFS_FILE)
        with open(pdfs_file, 'w', encoding='utf-8') as f:
            json.dump(pdfs_info, f, ensure_ascii=False, indent=2)
    
    def add_processed_pdf(self, collection_name: str, pdf_path: str, word_count: int):
        """Add a PDF to the list of processed PDFs for a collection."""
        pdfs_info = self.get_processed_pdfs(collection_name)
        
        # Calculate a hash of the file to track changes
        file_hash = ""
        try:
            with open(pdf_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
        except IOError:
            if RICH_AVAILABLE:
                console.print(f"[yellow]Warning: Could not read file {pdf_path} to calculate hash[/yellow]")
            else:
                logger.warning(f"Could not read file {pdf_path} to calculate hash")
        
        # Add file info
        pdfs_info[pdf_path] = {
            "path": pdf_path,
            "filename": os.path.basename(pdf_path),
            "hash": file_hash,
            "word_count": word_count,
            "processed_at": datetime.now().isoformat()
        }
        
        self.save_processed_pdfs(collection_name, pdfs_info)
    
    def extract_words_from_anki_package(self, apkg_path: str) -> Set[str]:
        """Extract the Turkish words from an existing Anki package."""
        if not os.path.exists(apkg_path) or not apkg_path.endswith('.apkg'):
            if RICH_AVAILABLE:
                console.print(f"[yellow]Warning: {apkg_path} is not a valid Anki package[/yellow]")
            else:
                logger.warning(f"{apkg_path} is not a valid Anki package")
            return set()
        
        words = set()
        temp_dir = None
        
        try:
            # Create a temporary directory to extract the package
            temp_dir = os.path.join(os.path.dirname(apkg_path), f"temp_anki_{int(time.time())}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Extract the package
            with zipfile.ZipFile(apkg_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Look for the database file
            db_file = os.path.join(temp_dir, "collection.anki2")
            if not os.path.exists(db_file):
                return words
            
            # We could use sqlite3 to extract data from the database
            # but for simplicity, we'll just check if there's a media file
            # with cards data in JSON format
            media_file = os.path.join(temp_dir, "media")
            if os.path.exists(media_file):
                try:
                    with open(media_file, 'r', encoding='utf-8') as f:
                        media_data = json.load(f)
                        # Process media references if needed
                except (json.JSONDecodeError, IOError):
                    pass
            
            # Check if there's a cards export file
            cards_file = None
            for file in os.listdir(temp_dir):
                if file.endswith('.json') and 'cards' in file.lower():
                    cards_file = os.path.join(temp_dir, file)
                    break
            
            if cards_file and os.path.exists(cards_file):
                try:
                    with open(cards_file, 'r', encoding='utf-8') as f:
                        cards_data = json.load(f)
                        # Extract Turkish words from the cards
                        if isinstance(cards_data, list):
                            for card in cards_data:
                                if isinstance(card, dict) and 'Turkish' in card:
                                    words.add(card['Turkish'].lower())
                except (json.JSONDecodeError, IOError):
                    pass
            
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"[red]Error extracting words from Anki package: {str(e)}[/red]")
            else:
                logger.error(f"Error extracting words from Anki package: {str(e)}")
        
        finally:
            # Clean up the temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        return words


class ProcessingState:
    """Class to maintain the processing state for resuming operations."""
    
    def __init__(self, output_dir: str, collection_name: str = ""):
        """Initialize the processing state."""
        self.pdf_paths: List[str] = []
        self.processed_pdfs: Set[str] = set()
        self.all_turkish_words: Set[str] = set()
        self.word_list: List[str] = []
        self.processed_words: List[str] = []
        self.translations: Dict[str, Dict] = {}
        self.output_dir: str = output_dir
        self.collection_name: str = collection_name or DEFAULT_COLLECTION
        self.start_time: float = time.time()
        self.excluded_words: Set[str] = set()  # Words to exclude (e.g., already in Anki)
        
    def save(self, state_file: str = STATE_FILE):
        """Save the current processing state to a file."""
        state_path = os.path.join(self.output_dir, state_file)
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(state_path, 'wb') as f:
            pickle.dump(self, f)
        
        if RICH_AVAILABLE:
            console.print(f"[green]Processing state saved to {state_path}[/green]")
        else:
            logger.info(f"Processing state saved to {state_path}")
    
    @staticmethod
    def load(output_dir: str, state_file: str = STATE_FILE) -> Optional['ProcessingState']:
        """Load the processing state from a file."""
        state_path = os.path.join(output_dir, state_file)
        
        if not os.path.exists(state_path):
            return None
        
        try:
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
                
            if RICH_AVAILABLE:
                console.print(f"[green]Loaded previous processing state from {state_path}[/green]")
            else:
                logger.info(f"Loaded previous processing state from {state_path}")
                
            return state
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"[red]Error loading state file: {e}[/red]")
            else:
                logger.error(f"Error loading state file: {e}")
            return None
    
    def remove_state_file(self, state_file: str = STATE_FILE):
        """Remove the state file after successful completion."""
        state_path = os.path.join(self.output_dir, state_file)
        if os.path.exists(state_path):
            os.remove(state_path)
            if RICH_AVAILABLE:
                console.print(f"[green]Removed state file {state_path}[/green]")
            else:
                logger.info(f"Removed state file {state_path}")

# Signal handler for graceful interruption
def signal_handler(sig, frame):
    """Handle interrupt signals by saving state before exit."""
    if RICH_AVAILABLE:
        console.print("\n[yellow]Received interrupt signal. Saving state before exit...[/yellow]")
    else:
        logger.info("\nReceived interrupt signal. Saving state before exit...")
    
    # The global state variable will be set in the process_pdfs function
    if 'current_state' in globals() and current_state is not None:
        current_state.save()
        if RICH_AVAILABLE:
            console.print("[green]State saved. You can resume processing later.[/green]")
        else:
            logger.info("State saved. You can resume processing later.")
    
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def process_pdfs(
    input_path: str, 
    output_dir: str, 
    api_key: str, 
    batch_size: int = 20, 
    resume: bool = True,
    collection_name: str = "",
    exclude_existing: bool = False,
    existing_anki_deck: str = "",
    description: str = ""
):
    """
    Process PDFs to extract Turkish words and create Anki cards.
    
    Args:
        input_path: Path to PDF file, directory of PDFs, or text file listing PDF paths
        output_dir: Base output directory for all collections
        api_key: Anthropic API key for Claude translations
        batch_size: Number of words to translate in each batch
        resume: Whether to resume from a previous run
        collection_name: Name of the collection to create or update
        exclude_existing: Whether to exclude words from an existing Anki deck
        existing_anki_deck: Path to an existing Anki deck to exclude words from
        description: Description of the collection
    """
    global current_state
    
    # Initialize collection manager
    collection_manager = CollectionManager(output_dir)
    
    # Set default collection name if not provided
    if not collection_name:
        collection_name = DEFAULT_COLLECTION
    
    # Get or create the collection directory
    collection_dir = collection_manager.create_or_update_collection(collection_name, description)
    
    # Load existing words to exclude if requested
    excluded_words = set()
    if exclude_existing and existing_anki_deck:
        if RICH_AVAILABLE:
            console.print(f"[blue]Loading existing words from {existing_anki_deck}...[/blue]")
        else:
            logger.info(f"Loading existing words from {existing_anki_deck}")
        
        excluded_words = collection_manager.extract_words_from_anki_package(existing_anki_deck)
        
        if RICH_AVAILABLE:
            console.print(f"[green]Found {len(excluded_words)} existing words to exclude[/green]")
        else:
            logger.info(f"Found {len(excluded_words)} existing words to exclude")
    
    # Check if we should resume from a previous state
    if resume:
        state = ProcessingState.load(collection_dir)
        if state and RICH_AVAILABLE:
            resume_confirmed = Confirm.ask(
                f"Previous processing state found for collection '{collection_name}'. Do you want to resume?",
                default=True
            )
            if not resume_confirmed:
                state = None
        elif state:
            logger.info(f"Previous processing state found for collection '{collection_name}'. Resuming...")
    else:
        state = None
    
    # Initialize a new state if not resuming
    if not state:
        state = ProcessingState(collection_dir, collection_name)
        state.excluded_words = excluded_words
        
        # Load previously processed PDFs from collection
        processed_pdfs_info = collection_manager.get_processed_pdfs(collection_name)
        state.processed_pdfs = set(processed_pdfs_info.keys())
        
        # Collect PDF paths
        if os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_path = os.path.join(root, file)
                        state.pdf_paths.append(pdf_path)
        elif os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
            state.pdf_paths.append(input_path)
        elif input_path.endswith('.txt'):
            # Assume it's a file with a list of PDF paths
            with open(input_path, 'r') as f:
                state.pdf_paths = [line.strip() for line in f if line.strip().lower().endswith('.pdf')]
        else:
            if RICH_AVAILABLE:
                console.print(f"[red]Invalid input: {input_path}. Please provide a PDF file, a directory containing PDFs, or a text file listing PDF paths.[/red]")
            else:
                logger.error(f"Invalid input: {input_path}. Please provide a PDF file, a directory containing PDFs, or a text file listing PDF paths.")
            return
        
        # Load existing translations for this collection
        translations_json = os.path.join(collection_dir, 'anki', f"{collection_name}.json")
        if os.path.exists(translations_json):
            try:
                with open(translations_json, 'r', encoding='utf-8') as f:
                    state.translations = json.load(f)
                if RICH_AVAILABLE:
                    console.print(f"[green]Loaded {len(state.translations)} existing translations for collection '{collection_name}'[/green]")
                else:
                    logger.info(f"Loaded {len(state.translations)} existing translations for collection '{collection_name}'")
            except (json.JSONDecodeError, IOError) as e:
                if RICH_AVAILABLE:
                    console.print(f"[yellow]Could not load existing translations: {str(e)}[/yellow]")
                else:
                    logger.warning(f"Could not load existing translations: {str(e)}")
    
    # Add excluded words to state if needed
    if excluded_words and not state.excluded_words:
        state.excluded_words = excluded_words
    
    # Set the current state for the signal handler
    current_state = state
    
    if not state.pdf_paths:
        if RICH_AVAILABLE:
            console.print("[red]No PDF files found.[/red]")
        else:
            logger.error("No PDF files found.")
        return
    
    if RICH_AVAILABLE:
        console.print(f"[blue]Found {len(state.pdf_paths)} PDF files to process for collection '{collection_name}'[/blue]")
    else:
        logger.info(f"Found {len(state.pdf_paths)} PDF files to process for collection '{collection_name}'")
    
    # Create collection directory
    os.makedirs(collection_dir, exist_ok=True)
    
    # Extract words from PDFs that haven't been processed yet
    pdfs_to_process = [pdf for pdf in state.pdf_paths if pdf not in state.processed_pdfs]
    
    # Dictionary to track words extracted from each PDF
    pdf_word_counts = {}
    
    if RICH_AVAILABLE and pdfs_to_process:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Extracting words from PDFs...", total=len(pdfs_to_process))
            
            for pdf_path in pdfs_to_process:
                progress.update(task, description=f"[cyan]Extracting from {os.path.basename(pdf_path)}[/cyan]")
                try:
                    words = extract_turkish_words(pdf_path)
                    
                    # Filter out excluded words
                    if state.excluded_words:
                        filtered_words = {w for w in words if w.lower() not in state.excluded_words}
                        if len(filtered_words) < len(words):
                            progress.print(f"[yellow]Excluded {len(words) - len(filtered_words)} words already in existing deck[/yellow]")
                        words = filtered_words
                    
                    # Save number of words found in this PDF
                    pdf_word_counts[pdf_path] = len(words)
                    
                    # Add to overall word set
                    state.all_turkish_words.update(words)
                    state.processed_pdfs.add(pdf_path)
                    
                    # Update collection tracking
                    collection_manager.add_processed_pdf(collection_name, pdf_path, len(words))
                    
                    progress.update(task, advance=1)
                    
                    # Save state periodically
                    state.save()
                    
                except Exception as e:
                    progress.print(f"[red]Error processing {pdf_path}: {e}[/red]")
    else:
        for pdf_path in pdfs_to_process:
            if pdf_path not in state.processed_pdfs:
                logger.info(f"Extracting words from {pdf_path}")
                try:
                    words = extract_turkish_words(pdf_path)
                    
                    # Filter out excluded words
                    if state.excluded_words:
                        filtered_words = {w for w in words if w.lower() not in state.excluded_words}
                        if len(filtered_words) < len(words):
                            logger.info(f"Excluded {len(words) - len(filtered_words)} words already in existing deck")
                        words = filtered_words
                    
                    # Save number of words found in this PDF
                    pdf_word_counts[pdf_path] = len(words)
                    
                    state.all_turkish_words.update(words)
                    state.processed_pdfs.add(pdf_path)
                    
                    # Update collection tracking
                    collection_manager.add_processed_pdf(collection_name, pdf_path, len(words))
                    
                    logger.info(f"Extracted {len(words)} potential Turkish words")
                    
                    # Save state periodically
                    state.save()
                    
                except Exception as e:
                    logger.error(f"Error processing {pdf_path}: {e}")
    
    if RICH_AVAILABLE:
        console.print(f"[green]Total unique Turkish words found in collection '{collection_name}': {len(state.all_turkish_words)}[/green]")
    else:
        logger.info(f"Total unique Turkish words found in collection '{collection_name}': {len(state.all_turkish_words)}")
    
    # Save the list of words for this collection
    words_list_path = os.path.join(collection_dir, f'{collection_name}_words.txt')
    with open(words_list_path, 'w', encoding='utf-8') as f:
        for word in sorted(state.all_turkish_words):
            f.write(f"{word}\n")
    
    if RICH_AVAILABLE:
        console.print(f"[green]Saved word list to {words_list_path}[/green]")
    else:
        logger.info(f"Saved word list to {words_list_path}")
    
    # Update word list if it's empty (first run or changed words)
    if not state.word_list or len(state.word_list) != len(state.all_turkish_words):
        state.word_list = list(state.all_turkish_words)
    
    # Filter out words that have already been processed
    words_to_translate = [w for w in state.word_list if w not in state.translations]
    
    if words_to_translate:
        if RICH_AVAILABLE:
            console.print(f"[blue]Translating {len(words_to_translate)} new words with batch size {batch_size}[/blue]")
        else:
            logger.info(f"Translating {len(words_to_translate)} new words with batch size {batch_size}")
        
        new_translations = translate_turkish_words(words_to_translate, api_key, batch_size, state)
        state.translations.update(new_translations)
        
        # Save after translations are complete
        state.save()
    else:
        if RICH_AVAILABLE:
            console.print("[yellow]No new words to translate[/yellow]")
        else:
            logger.info("No new words to translate")
    
    # Create Anki cards
    if state.translations:
        create_anki_deck(state.translations, collection_dir, collection_name)
    else:
        if RICH_AVAILABLE:
            console.print("[yellow]No translations available, cannot create Anki deck[/yellow]")
        else:
            logger.warning("No translations available, cannot create Anki deck")
    
    # Update collection statistics
    collection_manager.update_collection_stats(
        collection_name, 
        word_count=len(state.all_turkish_words),
        pdf_count=len(state.processed_pdfs)
    )
    
    # Clean up state file after successful completion
    state.remove_state_file()
    
    if RICH_AVAILABLE:
        elapsed_time = time.time() - state.start_time
        console.print(Panel.fit(
            f"[bold green]Processing completed successfully![/bold green]\n"
            f"Collection: [cyan]{collection_name}[/cyan]\n"
            f"Processed {len(pdfs_to_process)} new PDFs ({len(state.processed_pdfs)} total)\n"
            f"Extracted {len(state.all_turkish_words)} unique Turkish words\n"
            f"Created Anki deck with {len(state.translations)} cards\n"
            f"Total time: {elapsed_time:.1f} seconds",
            title="Collection Summary",
            border_style="green"
        ))
    else:
        elapsed_time = time.time() - state.start_time
        logger.info(f"Processing completed successfully for collection '{collection_name}' in {elapsed_time:.1f} seconds!")


def list_collections(output_dir: str):
    """List all available collections with their stats."""
    collection_manager = CollectionManager(output_dir)
    collections = collection_manager.list_collections()
    
    if not collections:
        if RICH_AVAILABLE:
            console.print("[yellow]No collections found[/yellow]")
        else:
            logger.info("No collections found")
        return
    
    if RICH_AVAILABLE:
        from rich.table import Table
        
        table = Table(title="Turkish Study Collections")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Words", justify="right")
        table.add_column("PDFs", justify="right")
        table.add_column("Last Updated", style="dim")
        
        for collection in collections:
            table.add_row(
                collection.get("name", ""),
                collection.get("description", ""),
                str(collection.get("word_count", 0)),
                str(collection.get("pdf_count", 0)),
                collection.get("last_updated", "").split("T")[0]
            )
        
        console.print(table)
    else:
        logger.info("Available collections:")
        for collection in collections:
            logger.info(f"- {collection.get('name')}: {collection.get('word_count')} words, {collection.get('pdf_count')} PDFs")


def list_collection_pdfs(output_dir: str, collection_name: str):
    """List all PDFs in a collection with their stats."""
    collection_manager = CollectionManager(output_dir)
    
    if not collection_manager.collection_exists(collection_name):
        if RICH_AVAILABLE:
            console.print(f"[red]Collection '{collection_name}' not found[/red]")
        else:
            logger.error(f"Collection '{collection_name}' not found")
        return
    
    pdfs_info = collection_manager.get_processed_pdfs(collection_name)
    if not pdfs_info:
        if RICH_AVAILABLE:
            console.print(f"[yellow]No PDFs found in collection '{collection_name}'[/yellow]")
        else:
            logger.info(f"No PDFs found in collection '{collection_name}'")
        return
    
    if RICH_AVAILABLE:
        from rich.table import Table
        
        table = Table(title=f"PDFs in Collection: {collection_name}")
        table.add_column("Filename", style="cyan")
        table.add_column("Words", justify="right")
        table.add_column("Processed Date", style="dim")
        
        for pdf_path, info in pdfs_info.items():
            table.add_row(
                info.get("filename", os.path.basename(pdf_path)),
                str(info.get("word_count", 0)),
                info.get("processed_at", "").split("T")[0]
            )
        
        console.print(table)
    else:
        logger.info(f"PDFs in collection '{collection_name}':")
        for pdf_path, info in pdfs_info.items():
            logger.info(f"- {info.get('filename')}: {info.get('word_count')} words")

def main():
    parser = argparse.ArgumentParser(description='Generate Anki cards from Turkish PDFs')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process PDFs and create Anki cards')
    process_parser.add_argument('input', help='PDF file, directory containing PDFs, or a text file with PDF paths')
    process_parser.add_argument('--output', '-o', default='output', help='Output directory')
    process_parser.add_argument('--collection', '-c', default='', help='Collection name (e.g., turkish_a1, beginner, etc.)')
    process_parser.add_argument('--description', '-d', default='', help='Description of the collection')
    process_parser.add_argument('--api-key', '-k', help='Anthropic API key')
    process_parser.add_argument('--batch-size', '-b', type=int, default=20, help='Batch size for translation requests')
    process_parser.add_argument('--no-resume', action='store_true', help='Do not resume from previous run')
    process_parser.add_argument('--model', default='claude-3-haiku-20240307', 
                            help='Claude model to use for translations (default: claude-3-haiku-20240307)')
    process_parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    process_parser.add_argument('--clean', action='store_true', help='Remove previous state file before starting')
    process_parser.add_argument('--exclude-from', help='Path to an existing Anki deck to exclude words from')
    
    # List collections command
    list_parser = subparsers.add_parser('list', help='List available collections')
    list_parser.add_argument('--output', '-o', default='output', help='Base output directory')
    
    # List PDFs in a collection command
    pdfs_parser = subparsers.add_parser('pdfs', help='List PDFs in a collection')
    pdfs_parser.add_argument('collection', help='Collection name')
    pdfs_parser.add_argument('--output', '-o', default='output', help='Base output directory')
    
    args = parser.parse_args()
    
    # Default to process command if none specified
    if not args.command:
        args.command = 'process'
        
    # Handle list command
    if args.command == 'list':
        list_collections(args.output)
        return
    
    # Handle pdfs command
    if args.command == 'pdfs':
        list_collection_pdfs(args.output, args.collection)
        return
    
    # Enable debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        if RICH_AVAILABLE:
            console.print("[yellow]Debug logging enabled[/yellow]")
        else:
            logger.info("Debug logging enabled")
    
    # Clean previous state if requested
    if args.clean and args.collection:
        collection_dir = os.path.join(args.output, args.collection)
        state_path = os.path.join(collection_dir, STATE_FILE)
        if os.path.exists(state_path):
            os.remove(state_path)
            if RICH_AVAILABLE:
                console.print(f"[yellow]Removed previous state file: {state_path}[/yellow]")
            else:
                logger.info(f"Removed previous state file: {state_path}")
    
    # Check for API key in environment if not provided as argument
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        if RICH_AVAILABLE:
            console.print("[red]Anthropic API key must be provided either as an argument or in the ANTHROPIC_API_KEY environment variable[/red]")
        else:
            logger.error("Anthropic API key must be provided either as an argument or in the ANTHROPIC_API_KEY environment variable")
        return
    
    # Show startup banner
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold blue]Turkish Study Assistant[/bold blue]\n"
            "Extract Turkish words from PDFs, translate them, and create Anki flashcards\n"
            f"Collection: [cyan]{args.collection or DEFAULT_COLLECTION}[/cyan] | "
            f"Batch size: [cyan]{args.batch_size}[/cyan] | "
            f"Resume: [cyan]{not args.no_resume}[/cyan]",
            title="🇹🇷 → 🇬🇧",
            border_style="blue"
        ))
    
    # Process PDFs
    try:
        process_pdfs(
            input_path=args.input,
            output_dir=args.output,
            api_key=api_key,
            batch_size=args.batch_size,
            resume=not args.no_resume,
            collection_name=args.collection,
            exclude_existing=bool(args.exclude_from),
            existing_anki_deck=args.exclude_from or "",
            description=args.description
        )
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n[yellow]Process interrupted by user.[/yellow]")
        else:
            logger.info("\nProcess interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()