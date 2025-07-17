
# Automated Book Publication Workflow

This project implements a streamlined workflow for publishing book content, featuring web scraping, AI-assisted rewriting, automated feedback generation, and version-controlled storage. Built with Python and Streamlit, it uses Playwright for scraping, NLTK and Transformers for text processing, and ChromaDB for persistent storage.

----------

## Overview

The **Automated Book Publication Workflow** automates the creation of polished book content from web sources in a single-pass process. It includes four key stages:

1.  **Collect Content**: Scrapes text and screenshots from a specified website.
    
2.  **AI Rewrites**: Uses AI to generate an initial rewritten version of the content.
    
3.  **Human Review**: Provides an AI-improved version of the content as feedback (no manual input, but we can edit the content at any stage and hit ctrl+enter to apply).
    
4.  **Save Final Version**: Saves the results with a unique version ID and stores them in ChromaDB.


## Project Structure

```
book-publication-workflow/
│
├── main.py                     (Main Python script containing the workflow logic)
│
├── output/                     (Directory for all generated files)
│   ├── chapter_text.txt        (Original text scraped from the website)
│   ├── chapter_screenshot.png  (Screenshot of the scraped webpage)
│   ├── rewritten_chapter.txt   (AI-rewritten text)   
│   ├── review_report.txt       (AI-Review report)    
│   └── feedback.txt            (AI-improved feedback)
│
├── chroma_db/                  (Directory for ChromaDB storage)
    └── (ChromaDB files)        (Automatically generated database files for text storage)

```

----------


## Prerequisites

-   **Python 3.8+**
    
-   Required libraries:
    
    -   streamlit
        
    -   playwright
        
    -   nltk
        
    -   textstat
        
    -   chromadb
        
    -   transformers
        
    -   torch
        
    -   asyncio
        

Install dependencies using:

```bash
pip install streamlit playwright nltk textstat chromadb transformers torch
playwright install
```

-   **NLTK Data**: Automatically downloaded on first run (requires internet access).
    
-   **Directories**: Create output and chroma_db folders in the project directory.
    
-   **Hardware**: Minimum 8GB RAM recommended for model loading and processing.
    

----------


## How to Run

Follow these steps to set up and execute the *Automated Book Publication Workflow* project on your local machine.

### Prerequisites

Ensure you have met the prerequisites listed in the Prerequisites section, including installing Python 3.8+, all required libraries, and creating the necessary directories.

### Step-by-Step Instructions

1.  **Navigate to the Project Directory**: Open a terminal or command prompt and change to the project directory where main.py is located:
    
    ```bash
    cd C:\~\book-publication-workflow
    ```
    
2.  **Activate a Virtual Environment (Optional but Recommended)**: To avoid conflicts with other Python projects, create and activate a virtual environment:
    
    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    ```
    
    If using a virtual environment, ensure you install dependencies within it (see Step 3).
    
3.  **Install Dependencies**: If not already installed, run the following command to install all required packages:
    
    ```bash
    pip install streamlit playwright nltk textstat chromadb transformers torch
    playwright install
    ```
    
    -   This downloads NLTK data and Playwright browser binaries, which may require internet access.
        
4.  **Prepare Directories**:
    
    -   Ensure the output and chroma_db directories exist. If not, create them manually:
        
        ```bash
        mkdir output 
        mkdir chroma_db
        ```
        

        
5.  **Launch the Streamlit Application**: Run the following command to start the Streamlit app:
    
    ```bash
    streamlit run main.py
    ```
    
    -   This opens a local web server, in your default web browser.
        
6.  **Configure the Workflow**:
    
    -   In the Streamlit UI, enter the URL (default : https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1).
        
    -   Enter the output directory path
        
    -   Enter the ChromaDB directory path
        
        
7.  **Execute the Pipeline**:
    
    -   Click the "Run Pipeline" button.
        
    -   The app will display a spinner while it scrapes the content, rewrites it, generates feedback, and saves the results. This process may take a few minutes depending on the URL and system performance.
        
8.  **View Results**:
    
    -   Once complete, the UI will show the output, including the version ID, file paths, and storage details in ChromaDB.
        
    -   Check the output directory for files like chapter_text.txt, rewritten_chapter.txt, review_report.txt, and feedback.txt.
        
    -   These files can be manually edited at any stage if needed.
        
9.  **Stop the Application**:
    
    -   Press Ctrl+C in the terminal to stop the Streamlit server when finished.
----------
## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request. For major changes, open an issue first to discuss.

----------

*Made with 🖤 by Ishwarya!*
