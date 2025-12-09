This repo includes backend documents.</br>


<h3>Basic Dependencies</h3>
• Operating System: Windows/macOS/Linux</br>
• Python 3.8+</br>
• npm 6.x+</br>
• Network Environment: Local service ports must be accessible</br>

<h3>Required Software</h3>
• Git (optional, for code cloning)</br>
• Web Browser (Chrome/Firefox latest versions recommended)</br>


<h2>Data Requirements</h2>

```python
├─/root/Project Folder
        ├─node_modules
        ├─backend
             ├─backend.py
             ├─backend_api.py
             ├─data folder
             ├─YU env
        ├─sciscinet-frontend
             ├─node_modules
                    ├─src
                       ├─App.tsx
                       ├─global.d.ts
                       ├─main.tsx
             ├─node_modules
             ├─public
             ├─index.html
        ├─YU env
...
```


<h2>Backend Application Deployment</h2>


<h3>Obtain Backend Code</h3>
Copy the backend code directory containing app.py to the target server:</br>

```python
cd backend  # Navigate to backend code directory
```

<h3>Install Dependencies</h3>

```python
# Create virtual environment
python -m venv venv
# macOS/Linux
source venv/bin/activate
```

<h4>Example from MacOS</h4>

```python
cd ../Desktop/../Yeshiva University Chatbot/backend
python -m venv YU
source YU/bin/activate
```

<h4>Install required packages:</h4>

```python
python -m pip install langchain-google-genai fastapi "uvicorn[standard]" pandas altair Ipython
```

<h3>Start Backend Environment</h3>

```python
python -m uvicorn backend_api:app --reload --port 8010
```

• The backend runs in: http://127.0.0.1:8010</br>
• For macOS, disable AirPlay to avoid port deprecation</br>

<h3>Data preprocessing</h3>
The main data for both projects are processed using the file: data_processing.ipynb</br>
Data are loaded from HuggingFace SciSciNet-v2 (please refer to: https://northwestern-cssi.github.io/sciscinet/)</br>
For the second project, data are filtered according to the project requirements and merged into a single large database, including key information such as paperid, author_id, fieldid, institution, citation, etc.</br>
