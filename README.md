# summarize_data_with_gen_ai_and_rag

- Create a chatbot that 



## Setting up the project
```
pip3 install virtualenv 
virtualenv my_env # create a virtual environment my_env
source my_env/bin/activate # activate my_env
```

The project is first retrieved 
```
# Clone the repo into a subfolder
git clone https://github.com/ibm-developer-skills-network/wbphl-build_own_chatbot_without_open_ai.git temp_clone

# Move its content into a new folder inside your project
mkdir build_chatbot_for_your_data
mv temp_clone/* build_chatbot_for_your_data
rm -rf temp_clone
```

# Install dependencies in the virtual environment
```
python -m pip install -r build_chatbot_for_your_data/requirements.txt
```

## Run the server

```python
python build_chatbot_for_your_data.server.py
```