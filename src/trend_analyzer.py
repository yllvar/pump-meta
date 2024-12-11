import spacy
from collections import Counter
import pandas as pd
import logging
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from telegram.helpers import escape_markdown

# Ensure spaCy model is downloaded: `python -m spacy download en_core_web_sm`
nlp = spacy.load("en_core_web_sm")


class TrendAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze(self, df):
        try:
            overall_trends = self.analyze_trends(df)
            keyword_insights = self.analyze_trends_from_descriptions(df)
            return {
                'overall_trends': overall_trends,
                'keyword_insights': keyword_insights
            }
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
            return {
                'overall_trends': "Error analyzing overall trends.",
                'keyword_insights': "Error analyzing keyword insights."
            }

    def analyze_trends(self, df):
        """Analyzes trends like top tokens, sentiment overview, and growth patterns."""
        if df.empty:
            return "No data available for analysis."

        try:
            all_text = self.prepare_text(df)
            top_keywords = self.extract_keywords(all_text)
            sentiment_analysis = self.analyze_sentiment(df)
            top_tokens = self.analyze_top_tokens(df)
            high_cap_analysis = self.analyze_high_cap_tokens(df)
            growth_analysis = self.analyze_growth(df)

            summary = self.format_summary(top_keywords, sentiment_analysis, top_tokens, high_cap_analysis, growth_analysis)
            return summary
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
            return "Error analyzing trends."
        
        

    def analyze_trends_from_descriptions(self, df):
        """Performs keyword extraction and trend analysis based on token descriptions."""
        if df.empty or 'Description' not in df.columns:
            return "No data available for description-based analysis."

        try:
            # Preprocess and tokenize descriptions
            descriptions = ' '.join(df['Description'].dropna().astype(str).tolist())
            processed_tokens = self.process_text_with_spacy(descriptions)
            keywords = Counter(processed_tokens).most_common(10)

            # Analyze keywords for trends
            keyword_insights = self.analyze_keywords(df, keywords)

            summary = "🔑 **Contextual Keyword Insights**\n\n"
            summary += '\n\n'.join(keyword_insights)
            return summary
        except Exception as e:
            self.logger.error(f"Error analyzing trends from descriptions: {e}")
            return "Error analyzing trends from descriptions."

    @staticmethod
    def prepare_text(df):
        """Prepares textual content from DataFrame columns."""
        text_columns = ['Name', 'Symbol', 'Description'] if 'Description' in df.columns else ['Name', 'Symbol']
        return ' '.join(df[text_columns].fillna('').astype(str).sum())

    @staticmethod
    def process_text_with_spacy(text):
        """Processes text using spaCy for tokenization and lemmatization."""
        doc = nlp(text)
        return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    @staticmethod
    def extract_keywords(text):
        """Extracts keywords using TF-IDF Vectorizer."""
        vectorizer = TfidfVectorizer(max_features=10)
        X = vectorizer.fit_transform([text])
        return vectorizer.get_feature_names_out()

    @staticmethod
    def analyze_sentiment(df):
        """Summarizes sentiment counts and examples."""
        if 'Sentiment Description' in df.columns:
            sentiment_counts = df['Sentiment Description'].value_counts().to_dict()
            sentiment_examples = {
                sentiment: ', '.join([
                    f"**{row['Symbol']}**" for _, row in df[df['Sentiment Description'] == sentiment].head(3).iterrows()
                ])
                for sentiment in sentiment_counts.keys()
            }
            return sentiment_counts, sentiment_examples
        return {}, {"Unknown": "N/A"}

    @staticmethod
    def analyze_top_tokens(df):
        """Finds the top tokens by Market Cap."""
        try:
            top_tokens = df[df['Market Cap'].notna()].nlargest(5, 'Market Cap')
            return [
                f"**{row['Symbol']}**: {row['Name']} - {row['Market Cap']:.2f} SOL\n"
                f"  📜 Contract: `{row['CA Address']}`\n"
                f"  💰 Price: {row.get('Price', 'N/A')} SOL\n"
                f"  🪙 Supply: {row.get('Supply', 'N/A')}"
                for _, row in top_tokens.iterrows()
            ]
        except Exception as e:
            logging.error(f"Error analyzing top tokens: {e}")
            return []

    @staticmethod
    def analyze_high_cap_tokens(df):
        """Analyzes high Market Cap tokens."""
        high_cap_tokens = df[df['Market Cap'].notna()]
        high_cap_avg = high_cap_tokens['Market Cap'].mean() if not high_cap_tokens.empty else 0
        high_cap_count = len(high_cap_tokens)
        return high_cap_avg, high_cap_count

    @staticmethod
    def analyze_growth(df):
        """Analyzes tokens with significant growth."""
        growth_threshold = 0.1
        significant_growth_tokens = df[df['Market Cap Change'] > growth_threshold]
        return [
            f"**{row['Symbol']}**: {row['Name']} - Growth: {row['Market Cap Change']:.2f} SOL\n"
            f"  📜 Contract: `{row['CA Address']}`\n"
            f"  💰 Price: {row.get('Price', 'N/A')} SOL\n"
            f"  🪙 Supply: {row.get('Supply', 'N/A')}"
            for _, row in significant_growth_tokens.iterrows()
        ]


    def analyze_keywords(self, df, keywords):
        """Analyzes keyword trends and aggregates meaningful insights."""
        keyword_stats = []
        for keyword, count in keywords:
            if not keyword.strip():  # Skip blank keywords
                continue

            tokens = df[df['Description'].str.contains(keyword, na=False, case=False)]
            avg_market_cap = tokens['Market Cap'].mean()
            avg_sentiment = tokens['Sentiment'].mean()

            keyword_stats.append({
                "keyword": escape_markdown(keyword),
                "occurrences": count,
                "avg_market_cap": avg_market_cap,
                "avg_sentiment": avg_sentiment,
            })

        # Process insights for improved formatting
        if not keyword_stats:
            return ["No significant keywords found for analysis."]

        # Sort keywords by occurrences and Market Cap
        keyword_stats = sorted(keyword_stats, key=lambda x: (-x["occurrences"], -x["avg_market_cap"]))

        # Generate summary and grouped insights
        summary = []
        summary.append("📊 **Summary:**")
        avg_sentiments = [stat["avg_sentiment"] for stat in keyword_stats if stat["avg_sentiment"] is not None]
        avg_market_caps = [stat["avg_market_cap"] for stat in keyword_stats if stat["avg_market_cap"] is not None]

        summary.append(f"- Most keywords have {'positive' if avg_sentiments and max(avg_sentiments) > 0 else 'neutral'} sentiment "
                    f"(Avg Sentiment: {sum(avg_sentiments)/len(avg_sentiments):.2f} if avg_sentiments else '0.00').")
        if avg_market_caps:
            summary.append(f"- The highest average Market Cap observed is {max(avg_market_caps):.2f} SOL.")

        summary.append("- Common keywords relate to themes like crypto, emotions, and community.\n")

        # Add grouped keywords
        grouped_keywords = ", ".join([stat["keyword"] for stat in keyword_stats if stat["avg_market_cap"] == max(avg_market_caps)])
        summary.append(f"🔹 **Keywords with Avg Market Cap: {max(avg_market_caps):.2f} SOL**\n   - {grouped_keywords}\n")

        # Highlight top keywords
        top_keyword = keyword_stats[0]
        summary.append(f"🔹 **Top Keyword by Occurrences:**")
        summary.append(f"   - Keyword: {top_keyword['keyword']}")
        summary.append(f"      - 📄 Occurrences: {top_keyword['occurrences']}")
        summary.append(f"      - 📈 Avg Market Cap: {top_keyword['avg_market_cap']:.2f} SOL")
        summary.append(f"      - 💬 Avg Sentiment: {top_keyword['avg_sentiment']:.2f}\n")

        # Add other keywords
        summary.append(f"🔹 **Other Keywords:**")
        other_keywords = ", ".join(stat["keyword"] for stat in keyword_stats[1:])
        summary.append(f"   - {other_keywords}")

        return summary

    def format_summary(self, top_keywords, sentiment_analysis, top_tokens, high_cap_analysis, growth_analysis):
        """Formats the summary of the analysis."""
        try:
            sentiment_counts, sentiment_examples = sentiment_analysis
            high_cap_avg, high_cap_count = high_cap_analysis

            summary = "📊 **Trend Analysis Summary**\n\n"
            summary += "🔑 **Top Keywords:**\n"
            summary += ', '.join(top_keywords) + "\n\n"

            summary += "📈 **Sentiment Overview:**\n"
            for sentiment, count in sentiment_counts.items():
                summary += f"- {sentiment}: {count} occurrences\n"
                summary += f"  Examples: {sentiment_examples[sentiment]}\n\n"

            summary += "💰 **Top Tokens by Market Cap:**\n"
            summary += '\n'.join(top_tokens) + "\n\n"

            summary += f"🏦 **High Market Cap Analysis:**\n"
            summary += f"- Average Market Cap: {high_cap_avg:.2f} SOL\n"
            summary += f"- Number of High Cap Tokens: {high_cap_count}\n\n"

            summary += "📊 **Growth Analysis:**\n"
            summary += '\n'.join(growth_analysis) + "\n"

            return summary
        except Exception as e:
            self.logger.error(f"Error formatting summary: {e}")
            return "Error formatting summary."

    
