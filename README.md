# HyperSketch Vector

A comprehensive system for generating, storing, and retrieving image descriptions using vector embeddings and AI models.

## 📁 Project Structure

```
hypersketch-vector/
├── main.py                       # Main entry point for retrieval demo
├── src/                          # Source code modules
│   ├── evaluation/               # Evaluation and testing
│   │   ├── evaluation.py         # Main evaluation logic and lambda sweep
│   │   └── generation_evaluation.py  # Generate evaluation datasets
│   ├── generation/               # Image description generation
│   │   └── generation.py         # AI-powered description generation
│   ├── retrieval/                # Vector operations and search
│   │   ├── arithmetic.py         # Embedding preprocessing and math
│   │   ├── insertion.py          # Vector database insertion
│   │   └── retrieval.py          # Search and retrieval logic
│   └── utils/                    # Shared utilities (empty)
├── data/                         # Data storage
│   ├── descriptions/             # Generated descriptions and evaluations
│   │   ├── descriptions.json     # AI-generated image descriptions
│   │   ├── evaluation.json       # Evaluation datasets
│   │   ├── picture_id.json       # Image ID mappings
│   │   └── test*.json            # Test data files
│   ├── images/                   # Image files for processing
│   └── results/                  # Analysis and evaluation results
│       └── lambda_sweep_*.csv    # Lambda parameter analysis results
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites

Ensure you have Python >=3.8 installed on your system.

### Installation

1. **Clone or download the project**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root with:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

### Quick Start

Run the main retrieval demo:
```bash
python main.py
```

This will process test queries and save results to `data/descriptions/test_results.json`.

## 📋 Individual Module Usage

### 1. Generate Image Descriptions
```bash
python src/generation/generation.py
```
- Processes images in `data/images/`
- Generates AI descriptions using GPT-5
- Saves results to `data/descriptions/descriptions.json`

### 2. Insert Vectors into Database
```bash
python src/retrieval/insertion.py
```
- Embeds descriptions using OpenAI text-embedding-3-small
- Uploads vectors to Pinecone database
- Handles deduplication automatically

### 3. Run Evaluations
```bash
python src/evaluation/evaluation.py
```
- Tests retrieval accuracy with lambda sweep (0.0 to 1.0)
- Generates comprehensive analysis reports
- Saves results to `data/results/`

### 4. Generate Evaluation Data
```bash
python src/evaluation/generation_evaluation.py
```
- Creates positive/negative/noisy paraphrases from descriptions
- Builds evaluation datasets for testing
- Uses concurrent processing for efficiency

## 🔧 Configuration

### Image Generation Settings
- **Model**: GPT-4o for vision analysis
- **Image Folder**: `data/images/`
- **Batch Size**: 10 images per progress update
- **Retry Logic**: 3 attempts with exponential backoff

### Vector Retrieval Settings  
- **Embedding Model**: OpenAI text-embedding-3-small (1536 dimensions)
- **Vector Database**: Pinecone with cosine similarity
- **Index Name**: `sref-test-index`
- **Lambda Values**: Configurable positive/negative prompt weighting (default: 0.9)

### Evaluation Settings
- **Test Pairs**: 6 pairs per picture (positive+negative, noisy+negative)
- **Lambda Sweep**: Tests values [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
- **Metrics**: Accuracy, rank distribution, pair type analysis

## 📊 Key Features

- **🎨 AI Description Generation**: Automated image analysis with detailed visual descriptors
- **🔍 Vector Search**: Efficient similarity search using embeddings
- **📈 Comprehensive Evaluation**: Lambda parameter optimization and accuracy testing
- **⚡ Concurrent Processing**: Multi-threaded operations for speed
- **🔄 Incremental Updates**: Smart deduplication and resumable operations
- **📁 Clean Structure**: Organized codebase with separation of concerns

## 🧪 System Workflow

1. **Image Processing**: AI analyzes images and generates descriptive text
2. **Vector Embedding**: Text descriptions are converted to 1536-dimensional vectors
3. **Database Storage**: Vectors are stored in Pinecone with metadata
4. **Query Processing**: User queries are embedded and matched against stored vectors
5. **Retrieval**: Most similar images are returned based on cosine similarity
6. **Evaluation**: System performance is tested with various parameter configurations

## 🛠️ Development

### File Organization

- **`src/retrieval/`**: Core vector operations and search functionality
- **`src/generation/`**: AI-powered image description generation
- **`src/evaluation/`**: Testing, analysis, and performance evaluation
- **`data/`**: All data files, images, and results

### Adding New Features

1. **New Retrieval Methods**: Modify `src/retrieval/retrieval.py`
2. **Enhanced Descriptions**: Update `src/generation/generation.py`
3. **Additional Evaluations**: Add to `src/evaluation/`
4. **Utility Functions**: Place in `src/utils/`

## 📄 Data Formats

- **descriptions.json**: `{filename: {id: str, path: str, description: [str]}}`
- **evaluation.json**: `{filename: [{paraphrased_prompt: str, paraphrase_type: str, picture_id: str}]}`
- **Results CSVs**: Pandas DataFrames with comprehensive analysis metrics

## 🚀 Running the Complete Pipeline

1. **Place images** in `data/images/`
2. **Generate descriptions**: `python src/generation/generation.py`
3. **Upload to database**: `python src/retrieval/insertion.py`
4. **Create evaluation data**: `python src/evaluation/generation_evaluation.py`
5. **Run evaluations**: `python src/evaluation/evaluation.py`
6. **Test retrieval**: `python main.py`


