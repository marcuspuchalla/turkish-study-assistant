# Turkish Study Assistant - Claude Guide

## Project Overview
Tool for extracting Turkish vocabulary from PDFs, translating with Claude AI, and creating Anki flashcards.

## Commands
- `python turkish_anki_generator.py process <input_path> --api-key <key>` - Process a PDF file or directory
- `python turkish_anki_generator.py list` - List available vocabulary collections
- `python turkish_anki_generator.py pdfs <collection_name>` - List PDFs in a collection
- `pip install -r requirements.txt` - Install dependencies

## Code Style Guidelines
- **Imports**: Group standard library, third-party, and local imports
- **Formatting**: Use 4 spaces for indentation, maximum line length of 100 characters
- **Typing**: Include type hints for function parameters and return values
- **Error Handling**: Use try/except blocks with specific exception types
- **Naming**: Use snake_case for variables/functions, PascalCase for classes
- **Documentation**: Include docstrings for functions and classes using """triple quotes"""

## Architecture
The system works in three stages: word extraction from PDFs, translation via Claude API, and Anki card generation.
Collections allow organizing vocabulary by level or topic (e.g., turkish_a1, turkish_a2).