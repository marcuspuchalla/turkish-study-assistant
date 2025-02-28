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
```

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.