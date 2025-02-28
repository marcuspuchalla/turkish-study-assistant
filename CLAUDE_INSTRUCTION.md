# Turkish Study Assistant - Comprehensive Development Guide

## Project Overview
The Turkish Study Assistant is a specialized tool designed to facilitate language learning through the automated creation of Anki flashcards from Turkish PDF documents. The system extracts Turkish vocabulary, leverages Claude AI for translations and example sentences, and generates structured Anki decks for effective learning.

## System Architecture

### Core Components
1. **PDF Processing Engine**: Extracts text from PDFs and identifies Turkish words
2. **Translation System**: Interfaces with Claude AI to translate words and generate examples
3. **Anki Card Generator**: Creates flashcard decks with templates and metadata
4. **Collection Management System**: Organizes vocabulary by level, topic, or source
5. **State Persistence Layer**: Handles interruption and resumption of processing

### Data Flow
```
PDF Documents → Text Extraction → Turkish Word Identification → 
Translation via Claude AI → Anki Card Generation → Collection Organization
```

### Key Files
- `turkish_anki_generator.py`: Main executable script containing all functionality
- `requirements.txt`: Dependencies specification
- `pyproject.toml`: Project configuration for modern Python tooling
- `README.md`: User documentation with usage examples
- `docs/images/`: Visual diagrams for documentation
  - `treasure_map_flowchart.svg`: Overall system workflow visualization
  - `state_persistence_map.svg`: Resumption functionality diagram
  - `anki_cards_chest.svg`: Visualization of the generated Anki cards

## Development History

### Initial Implementation
The project began with a focused objective: extract Turkish vocabulary from PDFs and create flashcards. The initial implementation included:
1. Basic PDF text extraction using PyMuPDF
2. Simple Turkish word detection heuristics based on character patterns
3. Claude API integration for translations
4. Minimal Anki card generation with genanki

### Evolution to Collection-Based System
As development progressed, we recognized the need for organizing vocabulary by level or topic, leading to:
1. Implementation of the `CollectionManager` class
2. Support for named collections (e.g., turkish_a1, turkish_a2)
3. Ability to incrementally add PDFs to collections
4. Collection statistics tracking

### Progress Tracking & Resumption
For handling large documents or interruptions, we added:
1. Automatic state persistence using Python's pickle module
2. Graceful interruption handling (signal handlers for SIGINT)
3. Batch processing for translations to avoid API limits
4. Progress tracking with rich terminal UI integration

### Visual Documentation
To improve understanding of the system's workflow, we created:
1. Treasure map style flowchart of the entire process
2. Visual guide for the resumption functionality
3. Anki cards treasure chest visualization

## Code Structure in Detail

### Major Classes

#### `ProcessingState` (Lines 732-794)
- Purpose: Maintains the state of processing for resumption
- Key attributes: 
  - `pdf_paths`: List of PDFs to process
  - `processed_pdfs`: Set of already processed PDFs
  - `all_turkish_words`: Set of extracted Turkish words
  - `translations`: Dictionary of translations
  - `excluded_words`: Words to exclude from processing
- Methods: `save()`, `load()`, `remove_state_file()`

#### `CollectionManager` (Lines 534-729)
- Purpose: Manages collections of vocabulary
- Key attributes:
  - `base_dir`: Base directory for collections
  - `collections_info`: Dictionary of collection metadata
- Methods: 
  - `create_or_update_collection()`
  - `update_collection_stats()`
  - `list_collections()`
  - `add_processed_pdf()`
  - `extract_words_from_anki_package()`

### Major Functions

#### `extract_turkish_words()` (Lines 101-158)
- Purpose: Extracts Turkish words from a PDF
- Input: PDF path
- Output: Set of Turkish words
- Implementation: 
  - Uses PyMuPDF to extract text
  - Cleans words with regex
  - Identifies Turkish words using heuristics
  - Provides progress UI when available

#### `translate_turkish_words()` (Lines 160-380)
- Purpose: Translates Turkish words to English
- Input: List of words, API key, batch size
- Output: Dictionary of translations
- Implementation:
  - Processes in batches to respect API limits
  - Creates a structured prompt for Claude
  - Extracts JSON from Claude's response
  - Handles errors and state saving

#### `create_anki_deck()` (Lines 382-532)
- Purpose: Creates Anki flashcards
- Input: Translations dictionary, output directory, deck name
- Output: Deck path
- Implementation:
  - Defines card templates with HTML/CSS styling
  - Creates notes with Turkish/English pairs
  - Generates .apkg file and JSON backup
  
#### `process_pdfs()` (Lines 817-1100)
- Purpose: Main processing function
- Input: Various parameters for processing
- Implementation:
  - Initializes collection manager
  - Loads/creates processing state
  - Extracts words from PDFs
  - Translates words
  - Creates Anki deck
  - Updates collection statistics

### CLI Implementation (Lines 1181-1286)
- Custom subcommands (process, list, pdfs)
- Argument parsing with argparse
- Environment variable support for API key

## Key Design Decisions

### 1. Single File Implementation
- Decision: Keep all code in a single file
- Rationale: Simplifies distribution and usage for non-technical users
- Trade-offs: Reduced modularity but improved portability

### 2. Progress Persistence
- Decision: Use pickle for state persistence
- Rationale: Simple serialization of Python objects
- Implementation: Regular state saving after batch processing

### 3. Turkish Word Identification
- Decision: Use character-based heuristics
- Rationale: Avoids dependency on complex NLP libraries
- Implementation: Checks for Turkish-specific characters and common word endings

### 4. Claude API Integration
- Decision: Use batch processing with structured prompts
- Rationale: Maximizes API efficiency while maintaining quality
- Implementation: JSON-formatted responses with word/example pairs

### 5. Rich UI Integration
- Decision: Optional dependency with fallback to standard logging
- Rationale: Improved UX when available without breaking core functionality
- Implementation: Conditional imports and alternative code paths

## Common Patterns and Idioms

### Conditional Rich UI
```python
if RICH_AVAILABLE:
    # Rich UI implementation
else:
    # Standard logging fallback
```

### Progress Tracking
```python
with Progress(...) as progress:
    task = progress.add_task(...)
    # Update progress during operations
```

### Error Handling
```python
try:
    # Operation that might fail
except SpecificException as e:
    # Graceful degradation with logging
```

### State Persistence
```python
# Save state after meaningful operations
state.save()

# Try to load previous state
state = ProcessingState.load(dir) or ProcessingState(dir)
```

## API Integration

### Claude API Usage
- Model: claude-3-haiku-20240307
- Temperature: 0 (deterministic outputs)
- Format: JSON structured responses
- System prompt: "You are a Turkish language expert, fluent in both Turkish and English."
- Batch size: Default 20 words per request (configurable)

### Anki Integration
- Package: genanki
- Note structure: Turkish word, English translation, Example, Example translation, Notes
- Templates: Two card types (Turkish→English, English→Turkish)
- Styling: Custom HTML/CSS for card appearance

## Best Practices Implemented

### 1. Progressive Enhancement
- Rich UI when available, fallback to standard logging
- Graceful handling of missing dependencies

### 2. Defensive Programming
- Extensive error handling for API calls
- Validation of user inputs
- Safeguards against data loss

### 3. User Experience
- Detailed progress information
- Color-coded outputs
- Automatic resumption of interrupted processing

### 4. Documentation
- Rich README with examples
- Visual diagrams of workflows
- Inline code documentation

## Potential Future Enhancements

### 1. Improved Turkish Detection
- More sophisticated linguistic rules
- Machine learning-based word classification
- Support for stemming and lemmatization

### 2. Enhanced Collection Management
- Hierarchical collections for progressive learning
- Vocabulary difficulty scoring
- Spaced repetition recommendations

### 3. Additional Output Formats
- Export to other flashcard systems
- PDF vocabulary lists
- Interactive web viewer

### 4. Audio Integration
- Text-to-speech for pronunciation
- Audio examples embedded in cards

### 5. UI Improvements
- Web interface or GUI application
- Visual vocabulary mapping
- Learning progress tracking

## Troubleshooting Guide

### API Issues
- Check for valid API key
- Verify API rate limits
- Examine debug logs for response parsing errors

### PDF Processing Problems
- Confirm PDF is text-based (not scanned images)
- Check encoding of PDF text
- Review character extraction in debug mode

### Collection Management
- Verify correct collection paths
- Check file permissions in output directories
- Review collection_info.json for corruption

## Command Reference

### Process Command
```bash
python turkish_anki_generator.py process <input_path> [options]
```
Options:
- `--output/-o`: Output directory
- `--collection/-c`: Collection name
- `--description/-d`: Collection description
- `--api-key/-k`: Claude API key
- `--batch-size/-b`: Batch size for translations
- `--no-resume`: Do not resume from previous run
- `--exclude-from`: Path to existing Anki deck for exclusion
- `--clean`: Remove previous state file
- `--debug`: Enable debug logging

### List Command
```bash
python turkish_anki_generator.py list [--output/-o <dir>]
```

### PDFs Command
```bash
python turkish_anki_generator.py pdfs <collection_name> [--output/-o <dir>]
```

## Development Environment Setup

### Dependencies
```bash
pip install pymupdf anthropic genanki rich
```

### API Key Configuration
```bash
export ANTHROPIC_API_KEY=<your-api-key>
```

## Testing Methodology

### PDF Processing Tests
- Small PDFs with known Turkish content
- PDFs with mixed Turkish/English text
- Edge cases: large PDFs, unusual formatting

### Translation Tests
- Verify batch processing works correctly
- Test error handling for API failures
- Confirm JSON parsing resilience

### Collection Management Tests
- Create, update, and list collections
- Test exclusion of words from existing decks
- Verify state persistence across runs