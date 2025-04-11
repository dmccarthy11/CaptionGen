import re
import emoji

def remove_header(text):
    """
    Cleans and extracts a caption if it contains a header with a colon, optional whitespace, then an opening quote. 
    If so, capture everything up to the closing quote
    """
    match = re.search(r':\s*\\?["\'](.*?)(\\?["\']|$)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return text.strip()
    
def remove_quotes(text):
    """
    Removes starting and ending quotes
    """
    return re.sub(r'^(["\'])(.*)\1$', r'\2', text)

def remove_emojis(text):
    """
    Removes all emojis and extra whitespaces
    """
    text = emoji.replace_emoji(text, replace='')

    return re.sub(r'\s{2,}', ' ', text).strip()

def remove_hashtags(text):
    """
    Removes all hashtags and extra whitespaces
    """
    text = re.sub(r'#\w+', '', text)

    return re.sub(r'\s{2,}', ' ', text).strip()

def remove_erraneous_chars(text):
    """
    Removes extra whitespace and empty hashtags
    """
    text = re.sub(r'#\w\s', '', text)

    return re.sub(r'\s{2,}', ' ', text).strip()
    
def clean_caption(text, emojis=True, hashtags=True):
    # Remove any header on the text
    text = remove_header(text)

    # Remove surrounding quotes
    text = remove_quotes(text)

    if not emojis:
        text = remove_emojis(text)

    if not hashtags:
        text = remove_hashtags(text)

    return text
