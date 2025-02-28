# Turkish Study Assistant

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

A Python tool to automatically create Anki flashcards from Turkish PDF documents by extracting words, translating them, and generating example sentences.

## 🚀 Features

- Extract Turkish vocabulary from PDF documents
- Distinguish between Turkish and English text
- Translate Turkish words to English using Claude AI
- Generate example sentences for each word
- Create ready-to-import Anki flashcards
- Supports both Turkish→English and English→Turkish card formats

## 📋 Requirements

- Python 3.6 or higher
- PyMuPDF (`pip install pymupdf`)
- Anthropic Python SDK (`pip install anthropic`)
- genanki (`pip install genanki`)
- An Anthropic API key for Claude

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/turkish-study-assistant.git
cd turkish-study-assistant

# Install required packages
pip install pymupdf anthropic genanki
```

## 🔧 Usage

```bash
# Process a single PDF file
./turkish_anki_generator.py path/to/turkish_document.pdf --api-key YOUR_API_KEY

# Process a directory of PDFs
./turkish_anki_generator.py path/to/pdf_directory/ --api-key YOUR_API_KEY

# Specify a custom output directory
./turkish_anki_generator.py path/to/pdf_directory/ --output path/to/output_dir --api-key YOUR_API_KEY

# You can also set the API key as an environment variable
export ANTHROPIC_API_KEY=YOUR_API_KEY
./turkish_anki_generator.py path/to/pdf_directory/

# Customize the batch size for translations (default: 20)
./turkish_anki_generator.py path/to/pdf_directory/ --batch-size 30

# Start fresh and ignore previous progress
./turkish_anki_generator.py path/to/pdf_directory/ --no-resume

# Force removal of previous state file
./turkish_anki_generator.py path/to/pdf_directory/ --clean

# Enable debug logging
./turkish_anki_generator.py path/to/pdf_directory/ --debug
```

### 🛑 Interrupting and Resuming

The script supports graceful interruption and resuming:

1. Press `Ctrl+C` at any time to pause processing
2. The current state will be automatically saved
3. Run the script again with the same output directory to resume where you left off
4. The script will automatically detect previous progress and ask if you want to resume

## 📚 Output

The script generates:

1. A text file with all extracted Turkish words
2. An Anki deck file (`.apkg`) that can be imported directly into Anki
3. A JSON file with all translations and examples for reference

The Anki cards include:
- The Turkish word
- The English translation
- An example sentence in Turkish
- The translation of the example sentence

## 🧠 How It Works

1. **Word Extraction**: The script analyzes PDF documents and extracts words that appear to be Turkish based on character patterns and common Turkish word endings.

2. **Translation**: Using Claude AI, the script translates each Turkish word to English and generates contextual example sentences.

3. **Card Generation**: The script creates Anki flashcards with both the translations and examples, formatted for effective learning.

## 🌟 Features in Detail

### Interactive Progress UI
- Real-time progress bars for all operations
- Detailed status updates during processing
- Sample translations displayed during processing
- Color-coded output for better readability

### State Persistence
- Automatic state saving after each batch of words
- Graceful handling of interruptions (Ctrl+C)
- Resume capability for long-running jobs
- Option to force a fresh start when needed

### Customization
- Configurable batch sizes for translation
- Timestamped output files to prevent overwrites
- Debug mode for troubleshooting
- Support for multiple input methods (file, directory, list)

### Anki Integration
- Beautifully styled flashcards
- Bidirectional learning (Turkish→English and English→Turkish)
- Notes field for personal annotations
- JSON export for backup and further processing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.