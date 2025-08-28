# Research Agent - AI-Powered Research with ML Analytics

A powerful Python application that combines AI-powered research capabilities with advanced machine learning text analysis using scikit-learn. This tool helps you conduct comprehensive research and analyze text data with professional visualizations.

## 🚀 Features

### 🤖 AI-Powered Research
- **Meta-Llama Integration**: Uses Meta-Llama/Llama-4-Maverick via OpenRouter API
- **Web Search**: Automated DuckDuckGo search with intelligent source selection
- **Content Analysis**: Advanced web page parsing and content extraction
- **Citation Management**: Automatic citation generation with source attribution
- **Report Generation**: Professional markdown reports with timestamps

### 📊 Machine Learning Analytics
- **Text Similarity Analysis**: TF-IDF vectorization with cosine similarity metrics
- **Document Clustering**: K-means clustering with PCA visualization
- **Statistical Insights**: Silhouette scores, inertia, and feature analysis
- **Professional Visualizations**: Matplotlib and Seaborn charts
- **Interactive Analysis**: Command-line interface for easy use

### 💻 Simple Local Execution
- **No Web Server Required**: Pure Python application
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Easy Installation**: Simple pip install process
- **VS Code Ready**: Includes configuration for development

## 📋 Requirements

- **Python 3.8+**
- **Internet Connection** (for API access and web search)
- **OpenRouter API Key** (free tier available)

## 🛠 Installation

### Step 1: Clone or Download
```bash
# If you have the project files
cd research-agent-simple

# Or create new directory and copy files
mkdir research-agent
cd research-agent
# Copy all files to this directory
```

### Step 2: Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### Step 3: Configure Environment
The `.env` file is already configured with your API key. If you need to update it:

```env
OPENROUTER_API_KEY=your-api-key-here
OPENROUTER_SITE_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_APP_NAME=Research Agent
```

## 🚀 Usage

### Interactive Mode (Recommended)
```bash
python research_agent.py --interactive
```

This launches an interactive menu where you can:
1. 🔍 Conduct AI-powered research
2. 📊 Analyze text similarity
3. 🎯 Cluster text documents
4. 📁 View generated outputs
5. ❌ Exit

### Command Line Usage

#### Research Query
```bash
python research_agent.py --query "What are the latest developments in renewable energy?"
```

#### Text Similarity Analysis
```bash
python research_agent.py --similarity "Machine learning is transforming healthcare" "AI is revolutionizing medical diagnosis" "Electric cars are becoming more popular"
```

#### Text Clustering
```bash
python research_agent.py --cluster "Climate change affects agriculture" "Global warming impacts farming" "AI improves medical diagnosis" "Machine learning helps doctors" --clusters 2
```

## 📊 Example Outputs

### Research Report Example
```markdown
# Research Report

**Query:** What are the latest developments in renewable energy?
**Generated:** 2024-12-28 14:30:22

---

## Recent Advances in Renewable Energy

The renewable energy sector has witnessed significant technological breakthroughs in 2024...

### Solar Technology Improvements
Recent developments in perovskite solar cells have achieved efficiency rates of over 25% [1]...

### Wind Energy Innovations
Offshore wind farms are now capable of generating power at costs competitive with fossil fuels [2]...

## Citations
[1] Nature Energy - "Perovskite solar cell efficiency breakthrough"
[2] International Energy Agency - "Offshore Wind Outlook 2024"
```

### Similarity Analysis Output
```
📊 Similarity Analysis Results:
   • Feature count: 1000
   • Top features: machine, learning, healthcare, medical, diagnosis

📊 Similarity Matrix:
   Text 1 ↔ Text 1: 1.000
   Text 1 ↔ Text 2: 0.847
   Text 1 ↔ Text 3: 0.234
   Text 2 ↔ Text 1: 0.847
   Text 2 ↔ Text 2: 1.000
   Text 2 ↔ Text 3: 0.198
```

### Clustering Results
```
🎯 Clustering Results:
   • Number of clusters: 2
   • Silhouette score: 0.742
   • Inertia: 0.156

📋 Cluster Assignments:
   Text 1: Cluster 0
   Text 2: Cluster 0
   Text 3: Cluster 1
   Text 4: Cluster 1
```

## 📁 Output Files

All generated files are saved in the `outputs/` directory:

- **Research Reports**: `research_report_YYYYMMDD_HHMMSS_query.md`
- **Similarity Heatmaps**: `similarity_heatmap_YYYYMMDD_HHMMSS.png`
- **Cluster Plots**: `cluster_scatter_YYYYMMDD_HHMMSS.png`

## 🔧 VS Code Integration

### Launch Configuration
Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Research Agent - Interactive",
            "type": "python",
            "request": "launch",
            "program": "research_agent.py",
            "args": ["--interactive"],
            "console": "integratedTerminal",
            "justMyCode": true
        },
        {
            "name": "Research Agent - Query",
            "type": "python",
            "request": "launch",
            "program": "research_agent.py",
            "args": ["--query", "${input:researchQuery}"],
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ],
    "inputs": [
        {
            "id": "researchQuery",
            "description": "Enter your research query",
            "default": "What are the latest trends in artificial intelligence?",
            "type": "promptString"
        }
    ]
}
```

### Workspace Settings
Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "python",
    "python.envFile": "${workspaceFolder}/.env",
    "python.terminal.activateEnvironment": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

## 🎨 Customization

### Modify AI Parameters
Edit the `ResearchAgent` class in `research_agent.py`:

```python
# Change AI model
self.model = "meta-llama/llama-3.1-8b-instruct:free"

# Adjust response parameters
response = self.client.chat.completions.create(
    model=self.model,
    temperature=0.5,  # Increase for more creative responses
    max_tokens=3000   # Increase for longer responses
)
```

### Customize ML Analysis
```python
# Modify TF-IDF parameters
self.vectorizer = TfidfVectorizer(
    max_features=2000,      # Increase vocabulary size
    ngram_range=(1, 2),     # Include bigrams
    min_df=2,               # Minimum document frequency
    max_df=0.8              # Maximum document frequency
)

# Adjust clustering parameters
kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=20,              # More initialization attempts
    max_iter=500            # More iterations
)
```

## 🔍 Troubleshooting

### Common Issues

#### API Key Error
```
ValueError: OPENROUTER_API_KEY is required in .env file
```
**Solution**: Ensure your `.env` file contains a valid OpenRouter API key.

#### Import Errors
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution**: Install requirements: `pip install -r requirements.txt`

#### Search Errors
```
Search error: [connection error]
```
**Solution**: Check your internet connection and try again.

#### Memory Issues
```
MemoryError during clustering
```
**Solution**: Reduce the number of texts or decrease `max_features` in TfidfVectorizer.

### Performance Tips

1. **Limit Text Length**: Keep individual texts under 1000 characters for better performance
2. **Reduce Features**: Lower `max_features` in TfidfVectorizer for faster processing
3. **Batch Processing**: Process large datasets in smaller batches
4. **Cache Results**: Save intermediate results to avoid reprocessing

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `python-dotenv` | >=1.0.1 | Environment variable management |
| `openai` | >=1.40.0 | OpenRouter API integration |
| `duckduckgo-search` | >=6.1.9 | Web search functionality |
| `beautifulsoup4` | >=4.12.3 | HTML parsing |
| `requests` | >=2.32.3 | HTTP requests |
| `scikit-learn` | >=1.3.0 | Machine learning algorithms |
| `numpy` | >=1.24.0 | Numerical computing |
| `pandas` | >=2.0.0 | Data manipulation |
| `matplotlib` | >=3.7.0 | Plotting and visualization |
| `seaborn` | >=0.12.0 | Statistical visualization |
| `plotly` | >=5.15.0 | Interactive plotting |

## 🎯 Use Cases

### Academic Research
- Literature reviews and source compilation
- Topic analysis and trend identification
- Citation management and bibliography creation

### Business Intelligence
- Market research and competitor analysis
- Customer feedback analysis and clustering
- Content similarity assessment

### Content Analysis
- Document classification and grouping
- Text similarity measurement
- Topic modeling and theme extraction

### Data Science Projects
- Text preprocessing and feature extraction
- Exploratory data analysis
- Prototype development and testing

## 🔮 Future Enhancements

### Planned Features
- **PDF Processing**: Direct PDF document analysis
- **Database Integration**: SQLite storage for results
- **Advanced Visualizations**: Interactive Plotly charts
- **Batch Processing**: Multiple query processing
- **Export Options**: CSV, JSON, and Excel export

### Possible Extensions
- **Custom Models**: Integration with local LLMs
- **Advanced NLP**: Named entity recognition and sentiment analysis
- **Web Interface**: Optional Flask web UI
- **API Mode**: REST API for integration with other tools

## 👨‍💻 Developer

**Developed by Abdul Haseeb**

A passionate software developer specializing in AI/ML applications and data science. This Research Agent demonstrates the power of combining modern AI APIs with traditional machine learning techniques for practical research and analysis tasks.

### Connect with Me

- **LinkedIn**: [hhttps://www.linkedin.com/in/abdul-haseeb-b1644b279/]
- **GitHub**: [hhttps://github.com/ahaseeb003](https://github.com/ahaseeb003)
- **Email**: [ahaseeb7300@gmail.com]



### Technical Skills Demonstrated

- **AI Integration**: OpenRouter API, OpenAI client library
- **Machine Learning**: scikit-learn, TF-IDF, K-means clustering, PCA
- **Data Visualization**: Matplotlib, Seaborn, Plotly
- **Web Scraping**: BeautifulSoup, requests, DuckDuckGo search
- **Python Development**: Object-oriented programming, command-line interfaces
- **Documentation**: Comprehensive user guides and code documentation

## 📄 License

This project is open source and available under the MIT License.

```
MIT License

Copyright (c) 2024 Abdul Haseeb

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

**Version**: 1.0.0  
**Last Updated**: December 28, 2024  
**Compatibility**: Python 3.8+, Cross-platform

🚀 **Ready to start researching? Run `python research_agent.py --interactive` to begin!**

