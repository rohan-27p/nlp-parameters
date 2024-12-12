from fastapi import FastAPI, HTTPException, Query
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from language_tool_python import LanguageTool
from pydantic import BaseModel
import nltk
import logging
from groq import Groq

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Download necessary NLTK data
nltk.download('punkt')

# Initialize FastAPI app and required tools
app = FastAPI()
tool = LanguageTool("en-US")
analyzer = SentimentIntensityAnalyzer()

# Initialize Groq client with API key
API_KEY = "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"
client = Groq(api_key=API_KEY)

# Helper function to calculate sentiment score
def get_sentiment_score(text: str):
    try:
        sentiment = analyzer.polarity_scores(text)
        compound_score = sentiment['compound']
        normalized_score = (compound_score + 1) / 2
        return round(normalized_score, 2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing sentiment: {e}")

@app.get("/sentiment")
def analyze_sentiment(text: str):
    score = get_sentiment_score(text)
    return {"sentiment_score": score}

# Helper function to calculate grammar severity
def calculate_severity(matches):
    severity = 0
    for match in matches:
        if "punctuation" in match.ruleId.lower():
            severity += 1
        elif "spelling" in match.ruleId.lower():
            severity += 0.7
        else:
            severity += 0.5
    return severity

# Helper function to calculate grammar score
def grammar_score(sentence: str):
    matches = tool.check(sentence)
    error_count = len(matches)
    severity = calculate_severity(matches)

    max_errors = 30
    base_score = max(0, 1 - (error_count / max_errors))
    severity_penalty = min(1, severity / 10)

    sentence_length = len(sentence.split())
    max_length = 50
    length_factor = min(1, sentence_length / max_length)

    score = base_score - severity_penalty
    score = score * length_factor

    return round(max(0, min(1, score)), 2)

@app.get("/grammar")
def analyze_grammar(text: str):
    gram_score = grammar_score(text)
    return {"grammar_score": gram_score}

# Helper function to calculate length score
def length_score(sent: str):
    min_length = 50
    max_length = 200
    length = len(sent.split())
    
    if length < min_length:
        score = length / min_length
    elif length > max_length:
        score = max(0, 1 - ((length - max_length) / max_length))
    else:
        score = 1/max(0, 1 - ((length - max_length) / max_length))  # Perfect score for sentences within the ideal length range
    
    return round(score, 2)

@app.get("/length")
def analyze_length(text: str = Query(..., description="Input text to analyze length score")):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty or whitespace.")
    
    try:
        len_score = length_score(text)
        return {"length_score": len_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
@app.get("/evaluate-review")
async def evaluate_review(
    place_type: str = Query(..., description="Type of place being reviewed (e.g., restaurant, movie theater)"),
    review_text: str = Query(..., description="Text of the review to be evaluated")
):
    try:
        # Construct the Groq API request payload
        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Evaluate the following review and provide a NUMERIC relevance score between 0 and 1.\n\n"
                        "The score should reflect how well the review matches the context of the place being reviewed, "
                        "considering the following criteria:\n\n"
                        "1. **Relevance**: The review should mention topics relevant to the place. For example:\n"
                        "   - If the place is a restaurant, the review should talk about food, service, ambiance, etc.\n"
                        "   - If the place is a movie theater, the review should talk about the movie experience, sound quality, seating, etc.\n"
                        "   - Reviews unrelated to the place's context (e.g., discussing a movie in a restaurant review) should receive a score near 0.\n\n"
                        "2. **Meaningfulness**: Avoid filler words, random characters, and gibberish (e.g., \"asdfjkl\" or repeated phrases).\n\n"
                        "3. **Coherence**: The review should be clear and make logical sense.\n\n"
                        "Provide a score between 0 and 1 (with decimals) based on these criteria. A score near 1 means the review is highly relevant and meaningful, "
                        "while a score near 0 means it is irrelevant, meaningless, or gibberish.\n\n"
                        f"*Context*: {place_type}\n\n"
                        f"*Review*: \"{review_text}\"\n\n"
                        "*Score*:"
                    )
                }
            ],
            temperature=0.5,
            max_tokens=1024,
            top_p=0.65,
            stream=False
        )

        # Extract the relevance score from the API response
        response_content = response.choices[0].message.content.strip()
        
        # Attempt to convert the response to a float
        try:
            relevance_score = float(response_content)
        except ValueError:
            # If direct conversion fails, try to extract a number
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', response_content)
            if match:
                relevance_score = float(match.group(1))
            else:
                raise ValueError("No numeric score found in the response")

        return {"relevance_score": min(max(relevance_score, 0), 1)}
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=f"Failed to parse relevance score: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating review: {e}")

@app.get("/analyze_review")
async def analyze_review(review_text: str, place_type: str):
    try:
        # Calculate individual scores
        sentiment_score = get_sentiment_score(review_text)
        grammar_score_value = grammar_score(review_text)

        # Extract relevance score
        relevance_result = await evaluate_review(place_type=place_type, review_text=review_text)
        relevance_score = relevance_result["relevance_score"]

        length_score_value = length_score(review_text)

        # Define weights
        weights = {
            "grammar": 0.3,
            "relevance": 0.4,
            "sentiment": 0.2,
            "length": 0.1
        }

        # Calculate weighted final score
        final_score = (
            (grammar_score_value * weights["grammar"]) +
            (relevance_score * weights["relevance"]) +
            (sentiment_score * weights["sentiment"]) +
            (length_score_value * weights["length"])
        )

        fake_threshold = 0.4
        is_fake = final_score < fake_threshold

        return {
            "status": "success",
            "final_score": round(final_score, 2),
            "is_fake": is_fake,
            "scores": {
                "grammar": round(grammar_score_value, 2),
                "relevance": round(relevance_score, 2),
                "sentiment": round(sentiment_score, 2),
                "length": round(length_score_value, 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing review: {e}")