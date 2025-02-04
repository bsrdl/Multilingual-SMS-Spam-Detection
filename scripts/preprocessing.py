import nltk 
from nltk.corpus import stopwords
from urllib.parse import urlparse
import re 
import fasttext

lang_model = fasttext.load_model("lid.176.bin")  # wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
nltk.download('stopwords')

# Function to extract meaningful parts of URLs
def extract_url_features(text):
    urls = re.findall(r'http[s]?://\S+', text)
    for url in urls:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace('www.', '').strip()  # Extract domain (e.g., ebebek.com)
        path_words = re.sub(r'[-_/]', ' ', parsed_url.path).strip()  # Extract words from path
        text = text.replace(url, f"URL TOKEN: {domain} {path_words}")
        text = re.sub(r'[^\w\s]', '', text).strip()
    return text

# Function to detect language of SMS messages
def detect_language(text):
    # Load fastText language detection model
    text = text.replace("\n", " ") 
    predictions = lang_model.predict([text], k=10)  # Get top 10 predictions
    
    # Extract language labels and probabilities
    labels = [label.replace("__label__", "") for label in predictions[0][0]]
    probabilities = predictions[1][0]
    
    # Filter only the allowed languages and get with the highest probability
    allowed_languages = {"en", "tr", "id"}
    best_language = "unknown"
    highest_prob = 0.0
    
    for label, prob in zip(labels, probabilities):
        if label in allowed_languages and prob > highest_prob:
            best_language = label
            highest_prob = prob
    return best_language


# Remove stopwords
def remove_stopwords(text, lang):
    if lang == 'en':
        stop_words = set(stopwords.words('english'))
    elif lang == 'tr':
        stop_words = set(stopwords.words('turkish'))
    elif lang == 'id':
        stop_words = set(stopwords.words('indonesian'))
    else:
        return text
    return ' '.join([word for word in text.split() if word not in stop_words])
        
# Preprocess dataset
def preprocess_data(df):
    df["Message"] = df.Message.str.lower()
    df = df.drop_duplicates()
    df["Message"] = df["Message"].apply(extract_url_features)
    df["Language"] = df["Message"].apply(detect_language)
    display(df.Language.value_counts())
    df = df[df.Language != 'unknown']
    df['Message'] = df.apply(lambda row: remove_stopwords(row['Message'], row['Language']), axis=1)
    df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})
    return df
    
