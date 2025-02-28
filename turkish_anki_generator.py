#!/usr/bin/env python3

import os
import sys
import argparse
import json
import re
import time
import signal
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any
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

class ProcessingState:
    """Class to maintain the processing state for resuming operations."""
    
    def __init__(self, output_dir: str):
        """Initialize the processing state."""
        self.pdf_paths: List[str] = []
        self.processed_pdfs: Set[str] = set()
        self.all_turkish_words: Set[str] = set()
        self.word_list: List[str] = []
        self.processed_words: List[str] = []
        self.translations: Dict[str, Dict] = {}
        self.output_dir: str = output_dir
        self.start_time: float = time.time()
        
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

def process_pdfs(input_path: str, output_dir: str, api_key: str, batch_size: int = 20, resume: bool = True):
    """Process all PDFs in the input path with resumable state."""
    global current_state
    
    # Check if we should resume from a previous state
    if resume:
        state = ProcessingState.load(output_dir)
        if state and RICH_AVAILABLE:
            resume_confirmed = Confirm.ask(
                "Previous processing state found. Do you want to resume?",
                default=True
            )
            if not resume_confirmed:
                state = None
        elif state:
            logger.info("Previous processing state found. Resuming...")
    else:
        state = None
    
    # Initialize a new state if not resuming
    if not state:
        state = ProcessingState(output_dir)
        
        # Collect PDF paths
        if os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        state.pdf_paths.append(os.path.join(root, file))
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
    
    # Set the current state for the signal handler
    current_state = state
    
    if not state.pdf_paths:
        if RICH_AVAILABLE:
            console.print("[red]No PDF files found.[/red]")
        else:
            logger.error("No PDF files found.")
        return
    
    if RICH_AVAILABLE:
        console.print(f"[blue]Found {len(state.pdf_paths)} PDF files to process[/blue]")
    else:
        logger.info(f"Found {len(state.pdf_paths)} PDF files to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract words from PDFs that haven't been processed yet
    pdfs_to_process = [pdf for pdf in state.pdf_paths if pdf not in state.processed_pdfs]
    
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
                    state.all_turkish_words.update(words)
                    state.processed_pdfs.add(pdf_path)
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
                    state.all_turkish_words.update(words)
                    state.processed_pdfs.add(pdf_path)
                    logger.info(f"Extracted {len(words)} potential Turkish words")
                    # Save state periodically
                    state.save()
                except Exception as e:
                    logger.error(f"Error processing {pdf_path}: {e}")
    
    if RICH_AVAILABLE:
        console.print(f"[green]Total unique Turkish words found: {len(state.all_turkish_words)}[/green]")
    else:
        logger.info(f"Total unique Turkish words found: {len(state.all_turkish_words)}")
    
    # Save the list of words
    words_list_path = os.path.join(output_dir, 'turkish_words.txt')
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
            console.print(f"[blue]Translating {len(words_to_translate)} words with batch size {batch_size}[/blue]")
        else:
            logger.info(f"Translating {len(words_to_translate)} words with batch size {batch_size}")
        
        new_translations = translate_turkish_words(words_to_translate, api_key, batch_size, state)
        state.translations.update(new_translations)
        # Save after translations are complete
        state.save()
    
    # Create Anki cards
    if state.translations:
        create_anki_deck(state.translations, output_dir)
    else:
        if RICH_AVAILABLE:
            console.print("[yellow]No translations were generated, cannot create Anki deck[/yellow]")
        else:
            logger.warning("No translations were generated, cannot create Anki deck")
    
    # Clean up state file after successful completion
    state.remove_state_file()
    
    if RICH_AVAILABLE:
        elapsed_time = time.time() - state.start_time
        console.print(Panel.fit(
            f"[bold green]Processing completed successfully![/bold green]\n"
            f"Processed {len(state.pdf_paths)} PDFs\n"
            f"Extracted {len(state.all_turkish_words)} unique Turkish words\n"
            f"Created Anki deck with {len(state.translations)} cards\n"
            f"Total time: {elapsed_time:.1f} seconds",
            title="Summary",
            border_style="green"
        ))
    else:
        elapsed_time = time.time() - state.start_time
        logger.info(f"Processing completed successfully in {elapsed_time:.1f} seconds!")

def main():
    parser = argparse.ArgumentParser(description='Generate Anki cards from Turkish PDFs')
    parser.add_argument('input', help='PDF file, directory containing PDFs, or a text file with PDF paths')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--api-key', '-k', help='Anthropic API key')
    parser.add_argument('--batch-size', '-b', type=int, default=20, help='Batch size for translation requests')
    parser.add_argument('--no-resume', action='store_true', help='Do not resume from previous run')
    parser.add_argument('--model', default='claude-3-haiku-20240307', 
                        help='Claude model to use for translations (default: claude-3-haiku-20240307)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--clean', action='store_true', help='Remove previous state file before starting')
    
    args = parser.parse_args()
    
    # Enable debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        if RICH_AVAILABLE:
            console.print("[yellow]Debug logging enabled[/yellow]")
        else:
            logger.info("Debug logging enabled")
    
    # Clean previous state if requested
    if args.clean:
        state_path = os.path.join(args.output, STATE_FILE)
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
            f"Using batch size: [cyan]{args.batch_size}[/cyan] | "
            f"Model: [cyan]{args.model}[/cyan] | "
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
            resume=not args.no_resume
        )
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n[yellow]Process interrupted by user.[/yellow]")
        else:
            logger.info("\nProcess interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()