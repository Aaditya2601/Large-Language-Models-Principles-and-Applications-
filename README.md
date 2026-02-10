
# Large Language Models Principles and Applications
Project: Automated Toxicity Classification via Prompt Engineering
***The development of an automated system which uses Large Language Models (LLMs) and advanced prompt engineering for online toxicity classification became part of my Master’s degree requirements. The main objective of this assignment involved testing the ability of LLMs to detect different levels of toxic content which include insults and threats and identity-based hate through zero-shot and few-shot learning methods.***

## My Approach
### **1. Data Preprocessing & Cleaning**
I worked with the Jigsaw Toxic Comment Classification dataset, which consists of approximately 159,000 Wikipedia comments. The online discourse environment needed me to develop a particular data cleaning system which used Python and Regex for processing. My focus was on:


1. The process involves eliminating duplicate characters and excessive punctuation which LLM tokenizers struggle to understand.

2. The model used standardization methods to prepare text information for emotional content evaluation but did not check the text formatting structure.

3. The model required multi-label output parsing to achieve correct identification of categories which could exist together (e.g. a comment that contains both obscene content and insulting language).

## **2. Prompt Engineering Strategy**
I developed two separate prompting methods to assess how well the model could reason through its responses.

**Zero-Shot**: I crafted precise system instructions to see if the model could categorize toxicity based solely on its internal knowledge.

**Few-Shot**: I selected particular "gold-standard" examples from the Jigsaw training set to help the model understand the distinction between "severe_toxic" and "toxic" labels.

***The main technical problem I needed to solve involved the model to produce class labels which it should output as JSON data so I used specific prompt boundaries to solve this issue.***

## **3. Technical Execution & Batch Processing**
The batch processing script which I created enables the system to handle the large dataset through its GPT-4 API interface. I focused on:

The system employs Response Parsing to identify API-generated string output labels which exist within the output.

The system includes error handling to manage API timeouts and rate limits which safeguards the entire evaluation dataset from losing any information.

## **4. Evaluation Metrics**
The system evaluation took place through a multi-label classification framework. The evaluation process for data involved methods which extended past the standard accuracy assessment procedures.

The evaluation process for toxic comment detection needs Precision and Recall and F1-Score metrics to achieve correct toxic comment identification and minimize false positive results.

I performed a complete assessment to find out which method between zero-shot and few-shot approaches provided the best monetary value for real-world moderation work.

## **Key Learning Outcomes**
***The project required me to use my Backend Technologies skills which included TypeScript/JavaScript programming to solve an essential social issue. The experience showed me how LLM-based automation systems work to replace or enhance conventional machine learning workflows through which I learned about the need for proper data cleaning and the skill required to create effective prompts which produce superior results.***

**Technologies Used:**
The system performs data cleaning through Python while using Regex functions and JavaScript/TypeScript for executing batch processing operations.

**Models**: GPT-4 (via API)

**Libraries**: Pandas, Scikit-learn (Evaluation Metrics)

The dataset used for this analysis comes from the Jigsaw Toxic Comment Classification competition which Kaggle and HuggingFace make available.
