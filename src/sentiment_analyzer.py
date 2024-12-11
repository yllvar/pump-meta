from textblob import TextBlob
import logging

class SentimentAnalyzer:
    @staticmethod
    def analyze_sentiment(text):
        try:
            if not text or not isinstance(text, str):
                return 0.0, "Unknown"
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            if polarity > 0.2:
                description = "Positive 😊"
            elif polarity < -0.2:
                description = "Negative 😢"
            else:
                description = "Neutral 😐"
            return polarity, description
        except Exception as e:
            logging.error(f"Error analyzing sentiment: {e}")
            return 0.0, "Unknown"

    def analyze(self, token_data):
        sentiment, sentiment_desc = self.analyze_sentiment(token_data.get('Description', ''))
        token_data['Sentiment'] = sentiment
        token_data['Sentiment Description'] = sentiment_desc
        return token_data