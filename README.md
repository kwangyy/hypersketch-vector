# HyperSketch Vector

A comprehensive system for generating, storing, and retrieving image descriptions using vector embeddings and AI models.

## 📁 Project Structure

```
hypersketch-vector/
├── src/                          # Source code modules
│   ├── evaluation/               # Evaluation and testing
│   │   ├── __init__.py
│   │   ├── evaluation.py         # Main evaluation logic and lambda sweep
│   │   └── generation_evaluation.py  # Generate evaluation datasets
│   ├── generation/               # Image description generation
│   │   ├── __init__.py
│   │   └── generation.py         # AI-powered description generation
│   ├── retrieval/                # Vector operations and search
│   │   ├── __init__.py
│   │   ├── arithmetic.py         # Embedding preprocessing and math
│   │   ├── insertion.py          # Vector database insertion
│   │   └── retrieval.py          # Search and retrieval logic
│   └── utils/                    # Shared utilities
│       └── __init__.py
├── data/                         # Data storage
│   ├── descriptions/             # Generated descriptions and evaluations
│   │   ├── descriptions.json     # AI-generated image descriptions
│   │   ├── evaluation.json       # Evaluation datasets
│   │   ├── picture_id.json       # Image ID mappings
│   │   └── test*.json            # Test data files
│   ├── images/                   # Image files (formerly "Kwang Yang SREF Tests")
│   └── results/                  # Analysis and evaluation results
│       ├── lambda_sweep_*.csv    # Lambda parameter analysis
│       └── *.csv                 # Other result files
├── main.py                       # Main entry point and demo
└── requirements.txt              # Python dependencies
```

## 🚀 Getting Started

### Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```bash
   # Create .env file with:
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

### Basic Usage

Run the main demo:
```bash
python main.py
```

## 📋 Module Usage

### 1. Generate Image Descriptions
```bash
python -m src.generation.generation
```
- Processes images in `data/images/`
- Generates AI descriptions using GPT-4
- Saves results to `data/descriptions/descriptions.json`

### 2. Insert Vectors into Database
```bash
python -m src.retrieval.insertion
```
- Embeds descriptions using OpenAI embeddings
- Uploads vectors to Pinecone database
- Handles deduplication automatically

### 3. Run Evaluations
```bash
python -m src.evaluation.evaluation
```
- Tests retrieval accuracy with different lambda values
- Generates comprehensive analysis reports
- Saves results to `data/results/`

### 4. Generate Evaluation Data
```bash
python -m src.evaluation.generation_evaluation
```
- Creates positive/negative/noisy paraphrases
- Builds evaluation datasets for testing
- Uses concurrent processing for efficiency

## 🔧 Configuration

### Generation Settings
- **Model**: GPT-4o for image analysis
- **Batch Size**: 10 images per progress update
- **Retry Logic**: 3 attempts with exponential backoff

### Retrieval Settings  
- **Embedding Model**: OpenAI text-embedding-3-small (1536 dimensions)
- **Vector Database**: Pinecone with cosine similarity
- **Lambda Values**: Configurable positive/negative prompt weighting

### Evaluation Settings
- **Test Pairs**: 6 pairs per picture (positive+negative, noisy+negative)
- **Lambda Sweep**: Tests values from 0.0 to 1.0
- **Metrics**: Accuracy, rank distribution, pair type analysis

## 📊 Key Features

- **🎨 AI Description Generation**: Automated image analysis with detailed descriptors
- **🔍 Vector Search**: Efficient similarity search using embeddings
- **📈 Comprehensive Evaluation**: Lambda parameter optimization and accuracy testing
- **⚡ Concurrent Processing**: Multi-threaded operations for speed
- **🔄 Incremental Updates**: Smart deduplication and resumable operations
- **📁 Organized Structure**: Clean separation of concerns

## 🧪 Evaluation Metrics

The system tracks several key metrics:
- **Overall Accuracy**: Percentage of correct retrievals
- **Rank Distribution**: Position of correct results in rankings  
- **Lambda Optimization**: Best positive/negative prompt weighting
- **Pair Type Analysis**: Performance by evaluation pair type

## 🛠️ Development

### Adding New Features

1. **New Evaluation Methods**: Add to `src/evaluation/`
2. **Generation Improvements**: Modify `src/generation/generation.py`
3. **Retrieval Algorithms**: Update `src/retrieval/`
4. **Utility Functions**: Place in `src/utils/`

### Running Individual Components

Each module can be run independently:
```python
from src.retrieval import retrieve, embed
from src.generation import generate_descriptions
from src.evaluation import run_evaluation, main_lambda_sweep
```

## 📄 File Formats

- **descriptions.json**: `{filename: {id, path, description: [list]}}`
- **evaluation.json**: `{filename: [{paraphrased_prompt, paraphrase_type, picture_id}]}`
- **Results CSVs**: Pandas DataFrames with analysis results

## 🔮 Future Enhancements

- [ ] Web interface for interactive querying
- [ ] Additional embedding models support
- [ ] Real-time vector database updates
- [ ] Advanced evaluation metrics
- [ ] Automated hyperparameter tuning
